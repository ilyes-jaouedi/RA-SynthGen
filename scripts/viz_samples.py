"""
Visualise N random samples from the EchoFusion test split.

For each sample:
  - extracts the camera image + Bartlett RA map live from RECORD@ sequences
  - loads the EchoFusion 3D bbox labels
  - builds the PSF x0 prior
  - saves a 4-panel figure:
      [camera image | x0 PSF prior | x1 GT RA map | x0 vs x1 overlay]

Also saves a standalone figure of the PSF kernel itself.

Usage (from RA-SynthGen root, with poetry shell active):
    python scripts/viz_samples.py
    python scripts/viz_samples.py --n-samples 8 --out-dir viz/samples
"""

import os, sys, argparse, random
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from data.extract_from_records import RadarProcessor
from models.psf_prior import build_psf_x0, load_psf_kernel
from data.dataset import normalise_ra, ELEV_IDX, LOG_MIN, LOG_MAX


# ── helpers ───────────────────────────────────────────────────────────────────

def load_labels(labels_dir: str, echo_frame_id: int) -> np.ndarray:
    """Load EchoFusion .txt labels -> (N, 7) float32 array, or empty (0,7)."""
    path = os.path.join(labels_dir, f"{echo_frame_id:06d}.txt")
    if not os.path.exists(path):
        return np.zeros((0, 7), dtype=np.float32)
    rows = []
    for line in open(path):
        parts = line.strip().split()
        if parts:
            vals = [float(v) for v in parts[:7]]
            rows.append(vals)
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 7), dtype=np.float32)


def ra_to_display(ra_512_751_11: np.ndarray) -> np.ndarray:
    """(512,751,11) -> (256,256) float in [0,1] for imshow."""
    ra2d = ra_512_751_11[:, :, ELEV_IDX]         # (512, 751)
    ra2d = cv2.resize(ra2d, (256, 256), interpolation=cv2.INTER_LINEAR)
    ra_norm = normalise_ra(ra2d)                  # [-1, 1]
    return ((ra_norm + 1) / 2).clip(0, 1)        # [0, 1]


def x0_to_display(x0_tensor) -> np.ndarray:
    """(1,256,256) torch tensor -> (256,256) float in [0,1]."""
    arr = x0_tensor[0].numpy()                    # [-1, 1]
    return ((arr + 1) / 2).clip(0, 1)


def img_to_display(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> RGB float [0,1] resized to 266×518."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (518, 266))
    return rgb.astype(np.float32) / 255.0


# ── PSF kernel figure ─────────────────────────────────────────────────────────

def save_psf_fig(psf_kernel: np.ndarray, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(psf_kernel, origin="lower", cmap="hot", aspect="auto")
    axes[0].set_title(f"2D PSF kernel  {psf_kernel.shape}", fontsize=11)
    axes[0].set_xlabel("Azimuth bins")
    axes[0].set_ylabel("Range bins")

    cx = psf_kernel.shape[1] // 2
    axes[1].plot(psf_kernel[psf_kernel.shape[0]//2, :], label="Azimuth slice")
    axes[1].plot(psf_kernel[:, cx], label="Range slice", linestyle="--")
    axes[1].set_title("PSF slices through centre", fontsize=11)
    axes[1].set_xlabel("Bin index")
    axes[1].set_ylabel("Normalised amplitude")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Analytic PSF kernel  (radial_psf_analytic.npy)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PSF kernel fig -> {out_path}")


# ── per-sample figure ─────────────────────────────────────────────────────────

def save_sample_fig(
    idx: int,
    echo_frame_id: int,
    img_display: np.ndarray,         # (266, 518, 3) float [0,1]
    x0_display: np.ndarray,          # (256, 256)    float [0,1]
    x1_display: np.ndarray,          # (256, 256)    float [0,1]
    bboxes: np.ndarray,              # (N, 7) or (0, 7)
    out_path: str,
):
    fig = plt.figure(figsize=(22, 6))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)

    # ── Panel 0: Camera ───────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img_display)
    ax0.set_title(f"Camera  (frame {echo_frame_id})", fontsize=10)
    ax0.axis("off")

    # ── Panel 1: x0 PSF prior ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(x0_display, origin="lower", cmap="inferno", vmin=0, vmax=1, aspect="auto")
    n_real = len(bboxes)
    ax1.set_title(f"x0  PSF prior  ({n_real} bbox)", fontsize=10)
    ax1.set_xlabel("Azimuth bin", fontsize=8)
    ax1.set_ylabel("Range bin",   fontsize=8)
    ax1.tick_params(labelsize=7)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # ── Panel 2: x1 GT RA map ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(x1_display, origin="lower", cmap="jet", vmin=0, vmax=1, aspect="auto")
    ax2.set_title("x1  GT Bartlett RA", fontsize=10)
    ax2.set_xlabel("Azimuth bin", fontsize=8)
    ax2.tick_params(labelsize=7)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # ── Panel 3: overlay (x0 in red, x1 in blue) ─────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    overlay = np.zeros((256, 256, 3), dtype=np.float32)
    overlay[:, :, 0] = x0_display                # red = PSF blobs
    overlay[:, :, 2] = x1_display                # blue = GT RA
    # green = where they coincide (bright spots should turn purple/magenta)
    overlay = overlay.clip(0, 1)
    ax3.imshow(overlay, origin="lower", aspect="auto")
    ax3.set_title("Overlay  (red=x0 | blue=x1)", fontsize=10)
    ax3.set_xlabel("Azimuth bin", fontsize=8)
    ax3.tick_params(labelsize=7)

    plt.suptitle(
        f"Sample {idx+1}  |  echo_frame_id={echo_frame_id}  |  {n_real} detected objects",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sample {idx+1:02d}  frame={echo_frame_id}  -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    # ── Validate required args ────────────────────────────────────────────────
    missing = [(n, v) for n, v in [
        ("--radial-root", args.radial_root),
        ("--eval-index",  args.eval_index),
        ("--eval-split",  args.eval_split),
        ("--labels-dir",  args.labels_dir),
    ] if v is None]
    if missing:
        for name, _ in missing:
            print(f"ERROR: {name} is required.")
        print("See README.md — Data Preparation section.")
        sys.exit(1)

    # ── Add DBReader to path ──────────────────────────────────────────────────
    radial_root = os.path.abspath(args.radial_root)
    dbreader_dir = os.path.join(radial_root, "DBReader")
    if dbreader_dir not in sys.path:
        sys.path.insert(0, dbreader_dir)
    parent = os.path.dirname(radial_root)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from DBReader.DBReader import SyncReader

    os.makedirs(args.out_dir, exist_ok=True)

    # ── PSF kernel ────────────────────────────────────────────────────────────
    psf_path = os.path.join(_ROOT, "assets", "radial_psf_calibrated.npy")
    print("Loading PSF kernel …")
    psf_kernel = load_psf_kernel(psf_path)
    print(f"  PSF kernel shape: {psf_kernel.shape}  max={psf_kernel.max():.3f}")
    save_psf_fig(psf_kernel, os.path.join(args.out_dir, "psf_kernel.png"))

    # ── Load eval index ───────────────────────────────────────────────────────
    df_idx   = pd.read_csv(args.eval_index)
    df_split = pd.read_csv(args.eval_split)
    test_ids = set(df_split.loc[df_split["split"] == "test", "echo_frame_id"].tolist())
    df_test  = df_idx[df_idx["echo_frame_id"].isin(test_ids)].copy()

    # Prefer frames that have label files
    labels_dir = args.labels_dir
    df_test["has_label"] = df_test["echo_frame_id"].apply(
        lambda fid: os.path.exists(os.path.join(labels_dir, f"{int(fid):06d}.txt"))
    )
    df_labelled = df_test[df_test["has_label"]]
    print(f"Test frames: {len(df_test)}  |  with labels: {len(df_labelled)}")

    # Sample N frames (prefer labelled)
    pool = df_labelled if len(df_labelled) >= args.n_samples else df_test
    sample_rows = pool.sample(n=min(args.n_samples, len(pool)), random_state=args.seed)
    sample_rows = sample_rows.sort_values("echo_frame_id").reset_index(drop=True)

    # ── Init RadarProcessor ───────────────────────────────────────────────────
    calib_path = os.path.join(radial_root, "SignalProcessing", "CalibrationTable.npy")
    print("Initialising RadarProcessor …")
    device = "cpu" if args.cpu else "cuda"
    processor = RadarProcessor(calib_path, device=device)

    # ── Open SyncReaders (one per sequence) ───────────────────────────────────
    seq_readers: dict[str, SyncReader] = {}

    # ── Extract and visualise ─────────────────────────────────────────────────
    print(f"\nExtracting {len(sample_rows)} sample frames …\n")

    for idx, row in sample_rows.iterrows():
        echo_id   = int(row["echo_frame_id"])
        seq_name  = row["seq_name"]
        local_idx = int(row["local_frame_idx"])

        if seq_name not in seq_readers:
            seq_path = os.path.join(radial_root, seq_name)
            try:
                seq_readers[seq_name] = SyncReader(seq_path, tolerance=20000, silent=True)
            except Exception as e:
                print(f"  Cannot open {seq_name}: {e}")
                continue

        db = seq_readers[seq_name]
        try:
            data = db.GetSensorData(local_idx)
        except Exception as e:
            print(f"  Frame {echo_id}: GetSensorData failed — {e}")
            continue

        cam_bgr   = data["camera"]["data"]
        adc_data  = {k: data[k] for k in ["radar_ch0", "radar_ch1", "radar_ch2", "radar_ch3"]}

        if cam_bgr is None:
            print(f"  Frame {echo_id}: no camera data")
            continue

        # Process radar
        try:
            ra_map = processor.process_frame(adc_data)  # (512, 751, 11)
        except Exception as e:
            print(f"  Frame {echo_id}: radar processing failed — {e}")
            continue

        # Load labels
        bboxes = load_labels(labels_dir, echo_id)       # (N, 7)

        # Build x0
        x0 = build_psf_x0(
            bboxes,
            psf_kernel,
            range_bins=256,
            az_bins=256,
            az_min_deg=-75.0,
            az_max_deg=75.0,
            r_max_m=103.0,
        )                                               # (1, 256, 256) tensor

        # Convert to display arrays
        img_disp = img_to_display(cam_bgr)
        x0_disp  = x0_to_display(x0)
        x1_disp  = ra_to_display(ra_map)

        out_path = os.path.join(args.out_dir, f"sample_{idx+1:02d}_frame{echo_id:06d}.png")
        n_in_loop = list(sample_rows.index).index(idx)
        save_sample_fig(n_in_loop, echo_id, img_disp, x0_disp, x1_disp, bboxes, out_path)

    print(f"\nAll figures saved to {args.out_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visualise RA-SynthGen dataset samples")
    p.add_argument("--n-samples",   type=int,  default=6)
    p.add_argument("--seed",        type=int,  default=7)
    p.add_argument("--out-dir",     default=os.path.join(_ROOT, "viz", "samples"))
    p.add_argument("--radial-root", default=None,
                   help="Path to RADIal/ folder (contains RECORD@* dirs). REQUIRED.")
    p.add_argument("--eval-index",  default=None,
                   help="Path to eval_index.csv. REQUIRED.")
    p.add_argument("--eval-split",  default=None,
                   help="Path to eval_split.csv. REQUIRED.")
    p.add_argument("--labels-dir",  default=None,
                   help="Path to EchoFusion labels_x/ directory. REQUIRED.")
    p.add_argument("--cpu",         action="store_true", help="Force CPU (no CUDA needed)")
    args = p.parse_args()
    main(args)
