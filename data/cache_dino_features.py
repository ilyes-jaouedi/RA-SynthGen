"""
Pre-compute and cache DINOv2 ViT-L/14 features for all frames.

The DINOv2 backbone is frozen during training, so there is no reason to
re-run it every step.  This script runs the encoder once over all frames
and saves (703, 1024) float16 arrays to

    <radial_root>/dino_features/{echo_frame_id:06d}.npy

Training then replaces the live SpatialEncoder forward pass with a
direct .npy load, shrinking per-batch time from ~54 s to < 1 s.

Usage (from RA-SynthGen root, with poetry shell active):
    python data/cache_dino_features.py
    python data/cache_dino_features.py --radial-root data/radial --batch-size 8
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import cv2
from torchvision import transforms
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from training.config import RADIAL_ROOT, DEVICE


# ── Image transform (must match dataset.py) ───────────────────────────────────

_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((266, 518)),
    transforms.ToTensor(),
    transforms.Normalize(_IMG_MEAN, _IMG_STD),
])


# ── DINOv2 encoder ────────────────────────────────────────────────────────────

def build_encoder(device):
    from transformers import Dinov2Model
    print("Loading DINOv2 ViT-L/14 …")
    enc = Dinov2Model.from_pretrained("facebook/dinov2-large")
    enc = enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    print(f"  DINOv2 loaded on {device}")
    return enc


@torch.no_grad()
def encode_batch(enc, imgs: list, device) -> np.ndarray:
    """
    imgs : list of (H, W, 3) uint8 BGR numpy arrays
    Returns (N, 703, 1024) float16 numpy array.
    """
    tensors = torch.stack([_transform(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                           for im in imgs]).to(device)
    out     = enc(pixel_values=tensors, interpolate_pos_encoding=True)
    feats   = out.last_hidden_state[:, 1:, :]   # drop CLS token → (N, 703, 1024)
    return feats.cpu().to(torch.float16).numpy()


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    radial_root = os.path.abspath(args.radial_root)
    cam_dir     = None
    for name in ("camera", "img", "images"):
        p = os.path.join(radial_root, name)
        if os.path.isdir(p):
            cam_dir = p
            break
    if cam_dir is None:
        raise FileNotFoundError(
            f"No camera/ folder found under {radial_root}. "
            "Run data/extract_from_records.py first."
        )

    out_dir = os.path.join(radial_root, "dino_features")
    os.makedirs(out_dir, exist_ok=True)

    # Collect all camera frames
    img_paths = sorted(
        glob.glob(os.path.join(cam_dir, "*.jpg"))
        + glob.glob(os.path.join(cam_dir, "*.jpeg"))
        + glob.glob(os.path.join(cam_dir, "*.png"))
    )
    print(f"Found {len(img_paths)} camera frames in {cam_dir}")

    # Filter frames that already have cached features
    if args.skip_existing:
        img_paths = [
            p for p in img_paths
            if not os.path.exists(
                os.path.join(out_dir,
                             os.path.splitext(os.path.basename(p))[0] + ".npy")
            )
        ]
        print(f"  {len(img_paths)} frames left after skipping existing")

    if not img_paths:
        print("Nothing to do.")
        return

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    enc = build_encoder(device)

    # Process in batches
    bs   = args.batch_size
    done = 0
    pbar = tqdm(range(0, len(img_paths), bs), desc="Caching DINOv2 features")
    for start in pbar:
        batch_paths = img_paths[start:start + bs]
        imgs        = [cv2.imread(p) for p in batch_paths]

        # Skip frames where imread failed
        valid = [(p, im) for p, im in zip(batch_paths, imgs) if im is not None]
        if not valid:
            continue
        batch_paths_ok, imgs_ok = zip(*valid)

        feats = encode_batch(enc, list(imgs_ok), device)   # (N, 703, 1024) fp16

        for feat, path in zip(feats, batch_paths_ok):
            fid      = os.path.splitext(os.path.basename(path))[0]   # e.g. "000018"
            out_path = os.path.join(out_dir, f"{fid}.npy")
            np.save(out_path, feat)

        done += len(valid)
        pbar.set_postfix({"saved": done})

    print(f"\nDone — {done} feature files written to {out_dir}/")
    print("Shape per file: (703, 1024)  dtype: float16")
    print("\nTo use cached features, set USE_CACHED_DINO=True in training/config.py")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cache DINOv2 features for all frames")
    p.add_argument("--radial-root",  default=RADIAL_ROOT,
                   help="Root folder produced by extract_from_records.py")
    p.add_argument("--batch-size",   type=int, default=8,
                   help="Images per DINOv2 forward pass (reduce if OOM)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip frames whose .npy feature file already exists")
    p.add_argument("--device",       default=None,
                   help="Override device (cuda / cpu). Default: auto-detect.")
    args = p.parse_args()
    main(args)
