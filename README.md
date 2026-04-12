# RA-SynthGen

**Synthetic Radar Range-Azimuth Map Generation via DiT + Conditional Flow Matching**

Generates physically realistic 256×256 RA maps conditioned on 3D object detections and camera images.  
Designed as a **data augmentation** pipeline for downstream radar detectors (e.g. EchoFusion).

---

## Core Idea

Standard RA maps are sparse and noisy. RA-SynthGen learns to **synthesise** plausible RA maps for any scene described by:

- A camera image (provides spatial context via DINOv2)
- 3D bounding boxes in radar frame (provide structural layout)

A calibrated PSF blob is placed at each detection's `(range, azimuth)` position to seed the ODE at a physically meaningful starting point, dramatically shortening the flow path and producing sharper results than starting from noise.

```
3D bbox labels  →  Calibrated PSF-blob x0 prior
Camera image    →  DINOv2 ViT-L/14 spatial tokens  ─┐
3D bbox params  →  BboxEncoder tokens               ─┤→ DiT cross-attention → velocity field
                                                     ┘
ODE integration:  x0 ──[v_θ]──→ x1_synthetic   (256×256 RA map)
```

---

## Architecture

| Component | Details |
|---|---|
| **DiT backbone** | 6 layers · 6 heads · 384 hidden dim · 16×16 patches on 256×256 |
| **Image encoder** | Frozen DINOv2 ViT-L/14 → (703, 1024) spatial tokens |
| **BboxEncoder** | Linear(7→128) → 2-layer Transformer → 1024-dim tokens per box |
| **Calibration bias** | Geometry soft-attention mask from `CalibrationTable.npy` (learnable scale) |
| **Bbox Gaussian bias** | Per-sample Gaussian centred at projected (range\_bin, az\_bin) of each detection |
| **PSF prior** | Calibrated PSF from Gram column of steering matrix; placed at each detection |
| **Training** | OT-CFM (straight-path flow matching) + TCR regularisation (λ=0.1) |
| **Inference** | Euler (fast) or `dopri5` ODE with classifier-free guidance (scale=2.0) |

---

## Required External Data

### RADIal dataset sequences

Download the **RADIal dataset** from the [official source](https://github.com/valeoai/RADIal).  
RA-SynthGen uses **6 specific RECORD@ sequences**:

```
RECORD@2020-11-21_11.54.31
RECORD@2020-11-21_11.58.53
RECORD@2020-11-22_12.03.47
RECORD@2020-11-22_12.25.47
RECORD@2020-11-22_12.45.05
RECORD@2020-11-22_12.54.38
```

Also required from the RADIal root:
```
RADIal/
├── SignalProcessing/
│   └── CalibrationTable.npy    ← steering vectors (required for PSF + beamforming)
├── DBReader/
│   └── DBReader.py             ← SyncReader class
├── RECORD@2020-11-21_11.54.31/ ← sequence folders listed above
└── ...
```

### EchoFusion annotations

You also need the **EchoFusion** annotation files:
```
EchoFusion_data/
├── labels_x/
│   └── {frame_id:06d}.txt      ← one file per frame, one box per line
├── eval_index.csv              ← maps echo_frame_id → seq_name + local_frame_idx
└── eval_split.csv              ← maps echo_frame_id → split (train/val/test)
```

---

## Setup

```bash
git clone https://github.com/<your-username>/RA-SynthGen.git
cd RA-SynthGen

# Install dependencies (Python 3.11–3.12 required)
poetry install

# Set the path to your EchoFusion labels directory
cp env.example .env
# Edit .env: set LABELS_DIR=/path/to/EchoFusion_data/labels_x

# Linux / macOS
source .env

# Windows PowerShell
Get-Content .env | ForEach-Object { $k,$v=$_.Split('=',2); [System.Environment]::SetEnvironmentVariable($k,$v) }
```

---

## Data Preparation

### Step 1 — Extract frames from RECORD@ sequences

```bash
poetry run python data/extract_from_records.py \
    --radial-root  /path/to/RADIal \
    --eval-index   /path/to/EchoFusion_data/eval_index.csv \
    --eval-split   /path/to/EchoFusion_data/eval_split.csv \
    --out-dir      data/radial \
    --skip-existing
```

This extracts 1035 frames from the 6 RECORD@ sequences into:
```
data/radial/
├── camera/       ← {frame_id:06d}.jpg  (camera frames)
├── radar_FFT/    ← {frame_id:06d}.npy  (512×751×11 Bartlett RA maps)
└── index.csv     ← frame manifest
```

Add `--cpu` to force CPU beamforming if no GPU is available (approx. 10× slower).

### Step 2 — (Optional) Regenerate analytic PSF

The calibrated PSF `assets/radial_psf_calibrated.npy` is included in this repo.  
The larger analytic PSF can be regenerated if needed:

```bash
poetry run python models/psf_prior.py \
    --calib /path/to/RADIal/SignalProcessing/CalibrationTable.npy \
    --out   assets/radial_psf_analytic.npy
```

### Step 3 — Verify the dataset

```bash
poetry run python data/dataset.py \
    --root       data/radial \
    --labels-dir /path/to/EchoFusion_data/labels_x \
    --psf        assets/radial_psf_calibrated.npy
```

---

## Training

```bash
# Cache DINOv2 features once (speeds up training from ~54 s/iter to ~6 s/iter)
# then start training:
.\run_cache_and_train.ps1          # Windows PowerShell

# Or run steps separately:
poetry run python data/cache_dino_features.py --skip-existing
poetry run python training/train.py
```

Checkpoints are saved to `weights/`:
- `ra_synthgen_best.pth` — lowest validation loss
- `ra_synthgen_last.pth` — latest epoch

Visualisation images (x0 PSF prior | generated x1 | GT x1 | camera) are saved to `viz/` every 5 epochs.

### Key hyperparameters (`training/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `HIDDEN_DIM` | 384 | DiT hidden dimension |
| `NUM_LAYERS` | 6 | DiT transformer depth |
| `BATCH_SIZE` | 16 | Reduce to 8 if GPU OOM |
| `EPOCHS` | 50 | Training epochs |
| `LAMBDA_TCR` | 0.1 | TCR regularisation weight |
| `CFG_DROPOUT_PROB` | 0.1 | CFG conditioning dropout rate |
| `USE_CACHED_DINO` | True | Use pre-cached DINOv2 features |

---

## Inference

### Single frame

```bash
poetry run python inference/generate.py \
    --checkpoint weights/ra_synthgen_best.pth \
    --image      data/radial/camera/001627.jpg \
    --labels     /path/to/EchoFusion_data/labels_x/001627.txt \
    --psf        assets/radial_psf_calibrated.npy \
    --out        generated_ra.png
```

### Batch augmentation

```bash
poetry run python inference/augment_dataset.py \
    --checkpoint weights/ra_synthgen_best.pth \
    --n-aug      3 \
    --out-dir    augmented_maps
```

### Visualise dataset samples (live extraction from RECORD@ sequences)

```bash
poetry run python scripts/viz_samples.py \
    --radial-root  /path/to/RADIal \
    --eval-index   /path/to/EchoFusion_data/eval_index.csv \
    --eval-split   /path/to/EchoFusion_data/eval_split.csv \
    --labels-dir   /path/to/EchoFusion_data/labels_x \
    --n-samples    6 \
    --out-dir      viz/samples
```

---

## Coordinate System

```
Radar frame:  X = forward,  Y = left,  Z = up
Azimuth:      -75° (right)  →  +75° (left)   →  751 raw bins → 256 after resize
Range:          0 m          →  103 m         →  512 raw bins → 256 after resize
Elevation:     -4°           →   +6°          →   11 slices (road-plane = index 5)
```

PSF blobs are placed at `(range_bin, az_bin)` derived from:
```python
r  = sqrt(x² + y²)           # range in metres
az = degrees(atan2(y, x))    # azimuth (positive = left of vehicle)
```

---

## Project Structure

```
RA-SynthGen/
├── assets/
│   ├── radial_psf_calibrated.npy    # Calibrated PSF from CalibrationTable (8 KB)
│   └── geometry_soft_bias.npy       # Calibration-aware attention mask (720 KB)
├── data/
│   ├── extract_from_records.py      # Extract camera + RA maps from RECORD@ sequences
│   ├── cache_dino_features.py       # Pre-compute DINOv2 features for all frames
│   ├── dataset.py                   # RADIalSynthDataset (with cached DINOv2 support)
│   └── radial/                      # Extracted data (gitignored, user-generated)
├── models/
│   ├── dit.py                       # DiT backbone + DINOv2 + calibration biases
│   ├── bbox_encoder.py              # 3D bbox → cross-attention tokens
│   └── psf_prior.py                 # Calibrated PSF-blob x0 constructor
├── training/
│   ├── config.py                    # All hyperparameters
│   ├── loss.py                      # OT-CFM loss + TCR regularisation
│   └── train.py                     # Training loop with validation and visualisation
├── inference/
│   ├── generate.py                  # Single-frame generation with CFG
│   └── augment_dataset.py           # Batch augmentation pipeline
├── scripts/
│   └── viz_samples.py               # Visualise dataset samples (4-panel figures)
├── weights/                         # Model checkpoints (gitignored)
├── viz/                             # Training visualisations (gitignored)
├── env.example                      # Template for environment variables
├── run_cache_and_train.ps1          # One-click: cache features + train (Windows)
└── pyproject.toml
```

---

## Ablations

Edit `training/config.py`:

| Ablation | Change |
|---|---|
| No bbox conditioning | `CFG_DROP_MODE = "bbox"`, `CFG_DROPOUT_PROB = 1.0` |
| No camera conditioning | `CFG_DROP_MODE = "clip"`, `CFG_DROPOUT_PROB = 1.0` |
| Gaussian noise x0 (no PSF) | Replace `build_psf_x0(...)` with `torch.randn_like(x1)` in `dataset.py` |
| No TCR regularisation | `LAMBDA_TCR = 0.0` |
| Larger model | `HIDDEN_DIM = 512`, `NUM_LAYERS = 8`, `NUM_HEADS = 8` |

---

## Relation to On-RADIal

This project is an independent spinout from [`On-RADIal/Project_RA_Gen_CFM`](https://github.com/Jaouedi/On-RADIal).

| Feature | Project_RA_Gen_CFM | RA-SynthGen |
|---|---|---|
| Image encoder | CLIP ViT-L/14 | **DINOv2 ViT-L/14** |
| Conditioning | Camera image only | Camera image + **3D bbox tokens** |
| x0 prior | YOLO 2D ray | **Calibrated PSF blobs at 3D detections** |
| Attention bias | Geometry soft mask | Geometry mask + **per-bbox Gaussian bias** |
| Ground truth | CMF-processed maps | Bartlett FFT maps (direct from ADC) |
| Purpose | Radar map generation | **Data augmentation** |
