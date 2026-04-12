"""
Central configuration for RA-SynthGen training.
Override any value here rather than editing train.py.
"""

import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root

RADIAL_ROOT   = os.path.join(_ROOT, "data", "radial")
# Primary label source: EchoFusion labels_x/ directory ({frame_id:06d}.txt per frame)
LABELS_DIR    = os.environ.get(
    "LABELS_DIR",
    os.path.join(_ROOT, "data", "labels_x"),   # default: labels_x/ inside data/
)
# Fallback CSV (set to None to use LABELS_DIR)
LABELS_CSV    = None
PSF_PATH      = os.path.join(_ROOT, "assets", "radial_psf_calibrated.npy")
WEIGHTS_DIR   = os.path.join(_ROOT, "weights")
VIZ_DIR       = os.path.join(_ROOT, "viz")

# ── Model ──────────────────────────────────────────────────────────────────────
RADAR_SIZE   = 256
PATCH_SIZE   = 16
HIDDEN_DIM   = 384
NUM_HEADS    = 6
NUM_LAYERS   = 6
MAX_BBOXES   = 8
CONTEXT_DIM  = 1024

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE       = 16          # reduce to 8 if GPU OOM
LR               = 1e-4
EPOCHS           = 50
LAMBDA_TCR       = 0.1         # weight for TCR regularisation term
CFG_DROPOUT_PROB = 0.1         # probability of dropping conditioning per batch
VAL_FRACTION     = 0.2

# Classifier-free guidance drop modes (used during training)
# 'both'  → drop CLIP + bbox with prob CFG_DROPOUT_PROB
# 'clip'  → drop only CLIP context
# 'bbox'  → drop only bbox context
CFG_DROP_MODE = "both"

# ── Feature caching ────────────────────────────────────────────────────────────
# Set True after running:  python data/cache_dino_features.py
# Loads pre-computed (703, 1024) float16 arrays from radial_root/dino_features/
# instead of running the frozen DINOv2 encoder live every step.
# Typical speedup: 54 s/it → ~3 s/it on a single GPU.
USE_CACHED_DINO = True     # flip to False to force live encoding

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── DataLoader ─────────────────────────────────────────────────────────────────
NUM_WORKERS = 4
PIN_MEMORY  = True

# ── Visualisation ──────────────────────────────────────────────────────────────
VIZ_EVERY_N_EPOCHS = 5        # save comparison images every N epochs

# ── ODE inference ──────────────────────────────────────────────────────────────
ODE_STEPS = 50                 # Euler steps for validation visualisation
ODE_METHOD = "dopri5"          # torchdiffeq method for generation
CFG_SCALE  = 2.0               # classifier-free guidance scale at inference
