"""
Batch augmentation pipeline.

Iterates over EchoFusion eval frames, generates a synthetic RA map for each,
and saves the result as a .npy file alongside the real data.

The output can be used directly as extra training samples for EchoFusion
(or any downstream radar detection model) by adding the generated maps to the
existing Dense_Dataset/radar_maps/ folder with an updated index CSV.

Usage:
    python inference/augment_dataset.py \\
        --checkpoint weights/ra_synthgen_best.pth \\
        --echo-root C:/path/to/EchoFusion_data \\
        --img-dir   C:/path/to/data/radial/img \\
        --labels-csv C:/path/to/RADIal/radial_bbox_labels.csv \\
        --psf       assets/radial_psf_analytic.npy \\
        --out-dir   augmented_maps \\
        --n-aug     3
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from torchvision import transforms

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from inference.generate  import load_image, load_labels_txt, generate
from models.psf_prior    import load_psf_kernel
from data.dataset        import IMG_MEAN, IMG_STD, LOG_MIN, LOG_MAX
from training.config     import MAX_BBOXES


def denormalise_ra(ra: np.ndarray) -> np.ndarray:
    """[-1,1] → power domain (inverse of dataset normalisation)."""
    ra = (ra + 1.0) / 2.0
    ra = ra * (LOG_MAX - LOG_MIN) + LOG_MIN
    return np.expm1(ra)


def augment(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load eval index ──────────────────────────────────────────────────
    index_path = os.path.join(args.echo_root, "eval_index.csv")
    df         = pd.read_csv(index_path)         # cols: echo_frame_id, seq_name, local_frame_idx
    labels_df  = pd.read_csv(args.labels_csv)
    labels_df.columns = [c.strip().lower() for c in labels_df.columns]
    col_map = {
        "numsample":  "frame_id",
        "radar_x_m":  "x", "radar_y_m": "y", "radar_z_m": "z",
        "dim_x_m":    "dim_x", "dim_y_m": "dim_y", "dim_z_m": "dim_z",
        "rotation_y": "theta",
    }
    labels_df = labels_df.rename(columns=col_map)
    labels_by_frame = labels_df.groupby("frame_id")

    psf_kernel = load_psf_kernel(args.psf)

    records = []  # for the output index CSV

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        fid     = int(row["echo_frame_id"])
        img_path = os.path.join(args.img_dir, f"{fid:06d}.jpg")
        if not os.path.exists(img_path):
            img_path = img_path.replace(".jpg", ".png")
        if not os.path.exists(img_path):
            continue

        # Build label tensor for this frame
        boxes_np = np.zeros((MAX_BBOXES, 7), dtype=np.float32)
        mask_np  = np.zeros(MAX_BBOXES, dtype=bool)
        if fid in labels_by_frame.groups:
            rows = labels_by_frame.get_group(fid)
            cols = [c for c in ["x","y","z","dim_x","dim_y","dim_z","theta"] if c in rows.columns]
            arr  = rows[cols].values.astype(np.float32)
            n    = min(len(arr), MAX_BBOXES)
            boxes_np[:n, :len(cols)] = arr[:n]
            mask_np[:n] = True

        bboxes    = torch.from_numpy(boxes_np).unsqueeze(0)   # (1,8,7)
        bbox_mask = torch.from_numpy(mask_np).unsqueeze(0)    # (1,8)
        image_t   = load_image(img_path)

        for aug_i in range(args.n_aug):
            x1_hat = generate(
                checkpoint=args.checkpoint,
                image_tensor=image_t,
                bboxes=bboxes,
                bbox_mask=bbox_mask,
                psf_kernel=psf_kernel,
                method=args.method,
                steps=args.steps,
            )
            # Convert to power domain and save
            ra_np    = x1_hat[0, 0].cpu().numpy()   # (256,256) in [-1,1]
            ra_power = denormalise_ra(ra_np)          # power domain
            out_name = f"synth_{fid:06d}_aug{aug_i:02d}.npy"
            out_path = os.path.join(args.out_dir, out_name)
            np.save(out_path, ra_power.astype(np.float32))
            records.append({
                "image":      f"{fid:06d}.jpg",
                "radar":      out_name,
                "source":     "synthetic",
                "frame_id":   fid,
                "aug_index":  aug_i,
            })

    # Save index CSV for downstream use
    index_out = os.path.join(args.out_dir, "synth_index.csv")
    pd.DataFrame(records).to_csv(index_out, index=False)
    print(f"\nGenerated {len(records)} synthetic RA maps → {args.out_dir}")
    print(f"Index saved → {index_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch synthetic RA augmentation")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--echo-root",   default=r"C:\Users\Ilyes\Desktop\On-RADIal\EchoFusion_data")
    p.add_argument("--img-dir",     default=r"C:\Users\Ilyes\Desktop\RA-SynthGen\data\radial\img")
    p.add_argument("--labels-csv",  default=r"C:\Users\Ilyes\Desktop\On-RADIal\RADIal\radial_bbox_labels.csv")
    p.add_argument("--psf",         default="assets/radial_psf_analytic.npy")
    p.add_argument("--out-dir",     default="augmented_maps")
    p.add_argument("--n-aug",       type=int,   default=3,      help="Augmentation multiplier per frame")
    p.add_argument("--method",      default="euler",            choices=["euler", "dopri5"])
    p.add_argument("--steps",       type=int,   default=50)
    args = p.parse_args()
    augment(args)
