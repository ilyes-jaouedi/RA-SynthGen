# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
poetry install        # install all dependencies
poetry shell          # activate virtualenv
```

Python 3.11–3.12 required. CUDA GPU strongly recommended (CPU works but beamforming is ~10× slower).

Set the `LABELS_DIR` environment variable to the EchoFusion `labels_x/` directory before training:
```bash
# Linux/Mac
export LABELS_DIR=/path/to/EchoFusion_data/labels_x
# Windows PowerShell
$env:LABELS_DIR = "C:\path\to\EchoFusion_data\labels_x"
```

## Key Commands

```bash
# Full pipeline (cache DINOv2 features once, then train)
.\run_cache_and_train.ps1                       # Windows PowerShell
poetry run python data/cache_dino_features.py --skip-existing  # step 1 only
poetry run python training/train.py                            # step 2 only

# Data extraction from RADIal RECORD@ sequences
poetry run python data/extract_from_records.py \
    --radial-root /path/to/RADIal \
    --eval-index  /path/to/EchoFusion_data/eval_index.csv \
    --eval-split  /path/to/EchoFusion_data/eval_split.csv \
    --skip-existing

# Inference
poetry run python inference/generate.py \
    --checkpoint weights/ra_synthgen_best.pth \
    --image data/radial/camera/001627.jpg \
    --labels /path/to/labels_x/001627.txt \
    --psf assets/radial_psf_calibrated.npy

# Quick sanity checks (run the module directly)
poetry run python data/dataset.py --root data/radial --labels-dir $LABELS_DIR --psf assets/radial_psf_calibrated.npy
poetry run python models/bbox_encoder.py
```

## Architecture Overview

The pipeline maps `(camera image, 3D bboxes) → synthetic 256×256 RA map` using **Optimal Transport Conditional Flow Matching (OT-CFM)** with a straight-path ODE between a PSF prior (x0) and the ground-truth Bartlett RA map (x1).

### Data flow

```
RECORD@ ADC data → RadarProcessor (Bartlett beamforming) → (512,751,11) RA map
                                                              ↓
EchoFusion labels (.txt)  →  build_psf_x0()  →  (1,256,256) PSF blob prior
Camera frame (.jpg)       →  DINOv2 ViT-L/14 →  (703,1024) spatial tokens  ──┐
3D bboxes (MAX_BBOXES,7)  →  BboxEncoder      →  (N,1024) tokens           ──┤
                                                                              ↓
x_t = (1-t)·x0 + t·x1  →  DiT(x_t, t, dino_ctx, bbox_ctx, bboxes, bbox_mask)  →  v_pred
```

### Model components

**`models/dit.py` — DiT**
- Patchifies the (1,256,256) RA map into 256 tokens of dim 384 (16×16 patches)
- adaLN modulation from sinusoidal time embedding
- Each `DiTBlock`: self-attention → cross-attention to context → MLP
- Context = `cat([dino_tokens, bbox_tokens], dim=1)` — shape `(B, 703+N_bbox, 1024)`
- Two calibration-aware attention biases stored as buffers:
  - `dino_attn_bias`: `(256, 703)` binary {-1, 0} mask × learnable `geo_mask_scale`; hard-blocks radar patches from attending to geometrically occluded image patches
  - Bbox Gaussian bias: computed live from raw `(x,y,z)` projected to patch grid; `bias = -dist²/(2σ²)` with learnable `geo_bbox_scale` (init=2.0 patches)
- CNN refinement head on the output before unpatchify
- `SpatialEncoder` wraps frozen DINOv2 ViT-L/14 and returns `last_hidden_state[:, 1:, :]` (no CLS)

**`models/psf_prior.py` — PSF prior**
- `compute_psf_from_calib()`: computes the real PSF from the Gram column at boresight of the calibrated steering matrix (`|G_w^H g_boresight|²`). Range PSF from real Hamming-windowed FFT mainlobe (not Gaussian).
- `build_psf_x0()`: stamps PSF at `(r_bin, az_bin)` for each bbox; amplitude ∝ `dim_x × dim_y / 9.4` (RCS proxy); returns `(1,256,256)` tensor in `[-1,1]`

**`models/bbox_encoder.py` — BboxEncoder**
- `Linear(7→128) → LayerNorm → ReLU → 2-layer TransformerEncoder → Linear(128→1024)`
- Padding tokens are zeroed after output projection
- `NullBboxContext`: trainable null embedding used when bbox condition is dropped (CFG)

**`training/loss.py` — CFMLoss**
- `v_target = x1 - x0` (constant velocity, OT straight path)
- `x1_pred = x_t + (1-t) * v_pred` for TCR
- `TCRLoss`: CFAR-style focal BCE on local `(μ, σ)` statistics — penalises failure to reproduce above-threshold returns

### DINOv2 feature caching

DINOv2 ViT-L is frozen and expensive (~50 s/iter if run live). `data/cache_dino_features.py` pre-computes `(703, 1024)` float16 arrays to `data/radial/dino_features/{frame_id:06d}.npy`. Set `USE_CACHED_DINO = True` in `training/config.py` (already the default) to load from disk instead.

### CFG dropout

During training, with probability `CFG_DROPOUT_PROB` (default 0.1), conditioning is dropped:
- `drop_clip=True`: DINOv2 context replaced by `null_dino_embed`, calibration bias zeroed
- `drop_bbox=True`: bbox context replaced by `NullBboxContext`, Gaussian bias mask zeroed
- `CFG_DROP_MODE = "both"` drops both simultaneously (default)

At inference, `euler_integrate()` / `odeint_integrate()` in `inference/generate.py` compute `v = v_uncond + CFG_SCALE * (v_cond - v_uncond)`.

### DiT forward signature

```python
model(x_t, t, dino_ctx, bbox_ctx, bboxes_raw, bbox_mask,
      drop_clip=False, drop_bbox=False)
# x_t         : (B, 1, 256, 256)
# t            : (B,)
# dino_ctx     : (B, 703, 1024)  — or cached float16 cast to float32 inside model
# bbox_ctx     : (B, N_bbox, 1024)
# bboxes_raw   : (B, MAX_BBOXES, 7)  — raw coords for Gaussian bias computation
# bbox_mask    : (B, MAX_BBOXES) bool
```

### Dataset

`data/dataset.py::RADIalSynthDataset` returns:
- `dino_feat`: `(703, 1024)` float32 when `use_cached_dino=True`; else `image`: `(3, 266, 518)` DINOv2-normalised (`IMG_MEAN=[0.485,0.456,0.406]`, `IMG_STD=[0.229,0.224,0.225]`)
- `x0`: `(1,256,256)` PSF blob prior in `[-1,1]`
- `x1`: `(1,256,256)` GT Bartlett RA map, log-normalised to `[-1,1]` (`LOG_MIN=15`, `LOG_MAX=23`)
- `bboxes`: `(MAX_BBOXES, 7)` zero-padded; `bbox_mask`: `(MAX_BBOXES,)` bool

### Radar geometry constants (used throughout)

- Azimuth: −75° to +75°, 751 raw bins → 256 after resize
- Range: 0–103 m, 512 raw bins → 256 after resize
- Elevation: 11 slices; road-plane = `ELEV_IDX = 5`
- Radar frame: X=forward, Y=left, Z=up

## Configuration

All hyperparameters live in `training/config.py`. Key entries:

| Variable | Default | Notes |
|---|---|---|
| `LABELS_DIR` | `$LABELS_DIR` env or `data/labels_x/` | EchoFusion annotation path |
| `USE_CACHED_DINO` | `True` | Flip to `False` to force live DINOv2 |
| `HIDDEN_DIM` | 384 | DiT hidden dim (512 for larger model) |
| `NUM_LAYERS` | 6 | DiT depth (8 for larger model) |
| `BATCH_SIZE` | 16 | Reduce to 8 if GPU OOM |
| `LAMBDA_TCR` | 0.1 | TCR regularisation weight |
| `CFG_SCALE` | 2.0 | Guidance scale at inference |
