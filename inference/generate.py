"""
RA-SynthGen generation (inference).

Given a camera image + 3D bounding boxes, generate a synthetic RA map.

Usage:
    python inference/generate.py \\
        --checkpoint weights/ra_synthgen_best.pth \\
        --image path/to/frame.jpg \\
        --labels path/to/frame_labels.txt \\
        --psf assets/radial_psf_analytic.npy \\
        --out output.png
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from models.dit          import DiT, SpatialEncoder
from models.bbox_encoder import BboxEncoder, NullBboxContext
from models.psf_prior    import build_psf_x0, load_psf_kernel
from data.dataset        import RADAR_SIZE, LOG_MIN, LOG_MAX, IMG_MEAN, IMG_STD
from training.config     import (
    CONTEXT_DIM, MAX_BBOXES, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS,
    ODE_STEPS, ODE_METHOD, CFG_SCALE, DEVICE,
)
from torchvision import transforms

try:
    from torchdiffeq import odeint
    HAS_ODE = True
except ImportError:
    HAS_ODE = False
    print("torchdiffeq not found — using Euler ODE solver.")


# ── Image preprocessing ───────────────────────────────────────────────────────

_img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((266, 518)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])


def load_image(path: str) -> torch.Tensor:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _img_transform(img).unsqueeze(0)   # (1, 3, 266, 518)


def load_labels_txt(path: str, max_bboxes: int = MAX_BBOXES):
    """
    Load a label file.  Each line: x y z dim_x dim_y dim_z theta [class]
    Returns (1, MAX_BBOXES, 7) tensor and (1, MAX_BBOXES) bool mask.
    """
    boxes = np.zeros((max_bboxes, 7), dtype=np.float32)
    mask  = np.zeros(max_bboxes,      dtype=bool)
    if path and os.path.exists(path):
        lines = [l.strip().split() for l in open(path) if l.strip()]
        for i, parts in enumerate(lines[:max_bboxes]):
            vals       = [float(v) for v in parts[:7]]
            boxes[i, :len(vals)] = vals
            mask[i]    = True
    return torch.from_numpy(boxes).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)


# ── ODE integration ───────────────────────────────────────────────────────────

def euler_integrate(
    model,
    x0: torch.Tensor,
    dino_ctx: torch.Tensor,
    bbox_ctx: torch.Tensor,
    bboxes_raw: torch.Tensor,   # (B, N, 7) raw 3D bbox coords for Gaussian bias
    bbox_mask: torch.Tensor,    # (B, N) bool
    steps: int,
    device,
) -> torch.Tensor:
    x = x0.clone()
    dt = 1.0 / steps
    # null tensors for unconditional pass
    null_dino = torch.zeros_like(dino_ctx)
    null_bbox = torch.zeros_like(bbox_ctx)
    null_mask = torch.zeros_like(bbox_mask)
    for i in range(steps):
        t = torch.tensor([i / steps], device=device).expand(x.shape[0])
        with torch.no_grad():
            v_cond   = model(x, t, dino_ctx, bbox_ctx, bboxes_raw, bbox_mask)
            v_uncond = model(x, t, null_dino, null_bbox, bboxes_raw, null_mask,
                             drop_clip=True, drop_bbox=True)
        v = v_uncond + CFG_SCALE * (v_cond - v_uncond)
        x = x + v * dt
    return x


def odeint_integrate(
    model,
    x0: torch.Tensor,
    dino_ctx: torch.Tensor,
    bbox_ctx: torch.Tensor,
    bboxes_raw: torch.Tensor,
    bbox_mask: torch.Tensor,
    method: str,
    device,
) -> torch.Tensor:
    null_dino = torch.zeros_like(dino_ctx)
    null_bbox = torch.zeros_like(bbox_ctx)
    null_mask = torch.zeros_like(bbox_mask)

    def ode_func(t, x):
        t_b = t.expand(x.shape[0])
        with torch.no_grad():
            v_cond   = model(x, t_b, dino_ctx, bbox_ctx, bboxes_raw, bbox_mask)
            v_uncond = model(x, t_b, null_dino, null_bbox, bboxes_raw, null_mask,
                             drop_clip=True, drop_bbox=True)
        return v_uncond + CFG_SCALE * (v_cond - v_uncond)

    t_span = torch.tensor([0.0, 1.0], device=device)
    traj   = odeint(ode_func, x0, t_span, method=method)
    return traj[-1]


# ── Main generation function ──────────────────────────────────────────────────

def generate(
    checkpoint:  str,
    image_tensor: torch.Tensor,    # (1, 3, 266, 518)
    bboxes:       torch.Tensor,    # (1, N, 7)
    bbox_mask:    torch.Tensor,    # (1, N)
    psf_kernel:   np.ndarray,
    method:       str = "euler",   # 'euler' | 'dopri5'
    steps:        int = ODE_STEPS,
) -> torch.Tensor:
    """
    Returns (1, 1, 256, 256) synthesised RA map in [-1, 1].
    """
    # ── Load model ─────────────────────────────────────────────────────────
    dino_enc  = SpatialEncoder().to(DEVICE).eval()
    bbox_enc  = BboxEncoder(context_dim=CONTEXT_DIM, max_bboxes=MAX_BBOXES).to(DEVICE).eval()
    model     = DiT(hidden=HIDDEN_DIM, depth=NUM_LAYERS, heads=NUM_HEADS,
                    context_dim=CONTEXT_DIM, max_bboxes=MAX_BBOXES).to(DEVICE).eval()

    ckpt = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    bbox_enc.load_state_dict(ckpt["bbox_enc"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    image_tensor = image_tensor.to(DEVICE)
    bboxes       = bboxes.to(DEVICE)
    bbox_mask    = bbox_mask.to(DEVICE)

    # ── Build x0 prior ─────────────────────────────────────────────────────
    real_boxes = bboxes[0][bbox_mask[0]].cpu().numpy()
    x0 = build_psf_x0(real_boxes, psf_kernel).unsqueeze(0).to(DEVICE)  # (1,1,256,256)

    # ── Encode conditions ──────────────────────────────────────────────────
    with torch.no_grad():
        dino_ctx = dino_enc(image_tensor)            # (1, N_dino, 1024)
        bbox_ctx = bbox_enc(bboxes, bbox_mask)       # (1, N_box,  1024)

    # ── Integrate ──────────────────────────────────────────────────────────
    if method == "euler" or not HAS_ODE:
        x1_hat = euler_integrate(
            model, x0, dino_ctx, bbox_ctx, bboxes, bbox_mask, steps, DEVICE)
    else:
        x1_hat = odeint_integrate(
            model, x0, dino_ctx, bbox_ctx, bboxes, bbox_mask, method, DEVICE)

    return x1_hat.clamp(-1.0, 1.0)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RA-SynthGen single-frame generation")
    p.add_argument("--checkpoint", required=True,          help="Path to .pth checkpoint")
    p.add_argument("--image",      required=True,          help="Path to camera image")
    p.add_argument("--labels",     default=None,           help="Path to label txt (x y z dim_x dim_y dim_z theta)")
    p.add_argument("--psf",        default="assets/radial_psf_analytic.npy")
    p.add_argument("--out",        default="generated_ra.png")
    p.add_argument("--method",     default="euler",        choices=["euler", "dopri5"])
    p.add_argument("--steps",      type=int, default=ODE_STEPS)
    args = p.parse_args()

    psf_kernel  = load_psf_kernel(args.psf)
    image_t     = load_image(args.image)
    bboxes, mask = load_labels_txt(args.labels, MAX_BBOXES)

    x1_hat = generate(
        checkpoint=args.checkpoint,
        image_tensor=image_t,
        bboxes=bboxes,
        bbox_mask=mask,
        psf_kernel=psf_kernel,
        method=args.method,
        steps=args.steps,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    ra_np = x1_hat[0, 0].cpu().numpy()  # (256, 256) in [-1, 1]
    ra_01 = (ra_np + 1) / 2             # [0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb);            axes[0].set_title("Camera image")
    axes[1].imshow(ra_01, cmap="jet", origin="lower", vmin=0, vmax=1)
    axes[1].set_title("Generated RA map")
    axes[1].set_xlabel("Azimuth bin"); axes[1].set_ylabel("Range bin")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")
