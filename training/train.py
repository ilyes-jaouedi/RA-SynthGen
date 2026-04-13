"""
RA-SynthGen training script.

Trains the DiT + BboxEncoder model using OT-CFM with PSF-blob x0 priors.

Usage:
    python training/train.py
    python training/train.py --radial-root data/radial --epochs 50
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Project imports ────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from data.dataset      import RADIalSynthDataset
from models.dit        import DiT, SpatialEncoder
from models.bbox_encoder import BboxEncoder, NullBboxContext
from training.loss     import CFMLoss
import training.config as C


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Visualisation helper ──────────────────────────────────────────────────────

def denorm(t: torch.Tensor) -> np.ndarray:
    """[-1,1] tensor → [0,1] numpy for imshow."""
    return ((t.squeeze().cpu().numpy() + 1) / 2).clip(0, 1)


_DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DINO_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def viz_batch(epoch, val_batch, model, dino_enc, bbox_enc, device, out_dir,
              radial_root=None, use_cached_dino=False):
    """Generate and save a comparison figure (x0, generated x1, GT x1, image)."""
    import cv2 as _cv2
    model.eval(); dino_enc.eval(); bbox_enc.eval()
    with torch.no_grad():
        x0        = val_batch["x0"].to(device)
        x1        = val_batch["x1"].to(device)
        bboxes    = val_batch["bboxes"].to(device)
        bbox_mask = val_batch["bbox_mask"].to(device)

        if use_cached_dino:
            dino_ctx = val_batch["dino_feat"][:1].to(device)   # already (1, 703, 1024)
        else:
            images   = val_batch["image"].to(device)
            dino_ctx = dino_enc(images[:1])

        bbox_ctx = bbox_enc(bboxes[:1], bbox_mask[:1])

        # Euler integration x0 → x1
        x_curr = x0[:1].clone()
        steps  = C.ODE_STEPS
        dt     = 1.0 / steps
        for i in range(steps):
            t_val = torch.tensor([i / steps], device=device)
            v = model(x_curr, t_val.expand(1), dino_ctx, bbox_ctx,
                      bboxes[:1], bbox_mask[:1])
            x_curr = x_curr + v * dt

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        # Camera image
        if use_cached_dino and radial_root is not None:
            fid      = int(val_batch["frame_id"][0])
            for ext in (".jpg", ".jpeg", ".png"):
                cpath = os.path.join(radial_root, "camera", f"{fid:06d}{ext}")
                if os.path.exists(cpath):
                    bgr = _cv2.imread(cpath)
                    im  = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    im  = _cv2.resize(im, (518, 266))
                    break
            else:
                im = np.zeros((266, 518, 3), dtype=np.float32)
        else:
            im = images[0].cpu().permute(1, 2, 0).numpy()
            im = (im * _DINO_STD + _DINO_MEAN).clip(0, 1)
        axes[0].imshow(im);                         axes[0].set_title("Camera")
        axes[1].imshow(denorm(x0[0]), cmap="jet", origin="lower", vmin=0, vmax=1)
        axes[1].set_title("x0  (PSF prior)")
        axes[2].imshow(denorm(x_curr[0]), cmap="jet", origin="lower", vmin=0, vmax=1)
        axes[2].set_title("Generated x1")
        axes[3].imshow(denorm(x1[0]), cmap="jet", origin="lower", vmin=0, vmax=1)
        axes[3].set_title("GT x1 (FFT data)")

        for ax in axes:
            ax.axis("off")

        plt.suptitle(f"Epoch {epoch}", fontsize=14)
        plt.tight_layout()
        path = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
        plt.savefig(path, dpi=120)
        plt.close()
    model.train(); dino_enc.train(); bbox_enc.train()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    set_seed(42)
    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.viz_dir,     exist_ok=True)

    print(f"Device: {C.DEVICE}")

    # ── Dataset ────────────────────────────────────────────────────────────
    if C.USE_CACHED_DINO:
        print("DINOv2 mode: CACHED features (fast)")
    else:
        print("DINOv2 mode: LIVE encoding (slow — run cache_dino_features.py first)")

    train_ds = RADIalSynthDataset(
        radial_root=args.radial_root,
        psf_path=args.psf_path,
        labels_dir=args.labels_dir,
        labels_csv=args.labels_csv,
        split="train",
        val_fraction=C.VAL_FRACTION,
        max_bboxes=C.MAX_BBOXES,
        use_cached_dino=C.USE_CACHED_DINO,
    )
    val_ds = RADIalSynthDataset(
        radial_root=args.radial_root,
        psf_path=args.psf_path,
        labels_dir=args.labels_dir,
        labels_csv=args.labels_csv,
        split="val",
        val_fraction=C.VAL_FRACTION,
        max_bboxes=C.MAX_BBOXES,
        use_cached_dino=C.USE_CACHED_DINO,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=C.NUM_WORKERS,
        pin_memory=C.PIN_MEMORY,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=C.NUM_WORKERS,
        pin_memory=C.PIN_MEMORY,
    )
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    # ── Models ─────────────────────────────────────────────────────────────
    # If using cached features, SpatialEncoder is only needed for viz_batch
    # (which still draws the raw image from val_batch["image"]).
    # When cached, we skip its forward pass entirely in the training loop.
    dino_enc  = SpatialEncoder().to(C.DEVICE).train()
    bbox_enc  = BboxEncoder(
        context_dim=C.CONTEXT_DIM,
        max_bboxes=C.MAX_BBOXES,
    ).to(C.DEVICE).train()
    null_bbox = NullBboxContext(C.MAX_BBOXES, C.CONTEXT_DIM).to(C.DEVICE)
    model     = DiT(
        hidden=C.HIDDEN_DIM,
        depth=C.NUM_LAYERS,
        heads=C.NUM_HEADS,
        context_dim=C.CONTEXT_DIM,
        max_bboxes=C.MAX_BBOXES,
    ).to(C.DEVICE).train()

    n_model = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    n_bbox  = sum(p.numel() for p in bbox_enc.parameters() if p.requires_grad) / 1e6
    print(f"DiT params     : {n_model:.2f}M")
    print(f"BboxEnc params : {n_bbox:.2f}M")
    # DINOv2 backbone is frozen

    # ── Loss ───────────────────────────────────────────────────────────────
    criterion = CFMLoss(lambda_tcr=args.lambda_tcr).to(C.DEVICE)

    # ── Optimiser ──────────────────────────────────────────────────────────
    # DINOv2 backbone is frozen; only train DiT + BboxEncoder + NullBbox
    trainable = (
        list(model.parameters())
        + list(bbox_enc.parameters())
        + list(null_bbox.parameters())
    )
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    best_val_loss = float("inf")
    start_epoch   = 0

    # ── Optional resume ────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.weights_dir, f"ra_synthgen_{C.MODEL_TAG}_last.pth")
    if os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=C.DEVICE)
        model.load_state_dict(ckpt["model"])
        bbox_enc.load_state_dict(ckpt["bbox_enc"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_epoch   = ckpt.get("epoch", 0) + 1
        print(f"Resumed at epoch {start_epoch}, best val {best_val_loss:.4f}")

    # ── Epoch loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train(); dino_enc.train(); bbox_enc.train()
        train_loss = train_mse = train_tcr = 0.0

        pbar = tqdm(train_dl, desc=f"Ep {epoch:03d}")
        for batch in pbar:
            images    = batch.get("image", None)
            if images is not None:
                images = images.to(C.DEVICE)
            x0        = batch["x0"].to(C.DEVICE)
            x1        = batch["x1"].to(C.DEVICE)
            bboxes    = batch["bboxes"].to(C.DEVICE)
            bbox_mask = batch["bbox_mask"].to(C.DEVICE)

            # ── Encode conditions ─────────────────────────────────────────
            if C.USE_CACHED_DINO:
                dino_ctx = batch["dino_feat"].to(C.DEVICE)      # (B, 703, 1024) pre-cached
            else:
                dino_ctx = dino_enc(images)                     # (B, N_dino, 1024) live
            bbox_ctx = bbox_enc(bboxes, bbox_mask)              # (B, N_box,  1024)

            # ── CFG dropout ───────────────────────────────────────────────
            drop_clip = drop_bbox = False
            r = random.random()
            if r < C.CFG_DROPOUT_PROB:
                if C.CFG_DROP_MODE == "both":
                    drop_clip = drop_bbox = True
                elif C.CFG_DROP_MODE == "clip":
                    drop_clip = True
                elif C.CFG_DROP_MODE == "bbox":
                    drop_bbox = True

            if drop_bbox:
                bbox_ctx = null_bbox.expand(x0.shape[0])

            # ── Sample t ─────────────────────────────────────────────────
            t = torch.rand(x1.shape[0], device=C.DEVICE)

            # ── Forward ───────────────────────────────────────────────────
            # Build x_t and get v_pred
            t_view  = t.view(-1, 1, 1, 1)
            x_t     = (1 - t_view) * x0 + t_view * x1
            v_pred  = model(x_t, t, dino_ctx, bbox_ctx, bboxes, bbox_mask,
                            drop_clip=drop_clip, drop_bbox=drop_bbox)

            # ── Loss ──────────────────────────────────────────────────────
            loss, l_mse, l_tcr = criterion(v_pred, x0, x1, t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_mse  += l_mse.item()
            train_tcr  += l_tcr.item()
            pbar.set_postfix({"mse": f"{l_mse.item():.4f}", "tcr": f"{l_tcr.item():.4f}"})

        scheduler.step()

        n = len(train_dl)
        avg_train = train_loss / n
        avg_mse   = train_mse  / n
        avg_tcr   = train_tcr  / n

        # ── Validation ────────────────────────────────────────────────────
        model.eval(); dino_enc.eval(); bbox_enc.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                images    = batch.get("image", None)
                if images is not None:
                    images = images.to(C.DEVICE)
                x0        = batch["x0"].to(C.DEVICE)
                x1        = batch["x1"].to(C.DEVICE)
                bboxes    = batch["bboxes"].to(C.DEVICE)
                bbox_mask = batch["bbox_mask"].to(C.DEVICE)

                if C.USE_CACHED_DINO:
                    dino_ctx = batch["dino_feat"].to(C.DEVICE)
                else:
                    dino_ctx = dino_enc(images)
                bbox_ctx = bbox_enc(bboxes, bbox_mask)
                t        = torch.rand(x1.shape[0], device=C.DEVICE)
                t_view   = t.view(-1, 1, 1, 1)
                x_t      = (1 - t_view) * x0 + t_view * x1
                v_pred   = model(x_t, t, dino_ctx, bbox_ctx, bboxes, bbox_mask)
                loss, _, _ = criterion(v_pred, x0, x1, t)
                val_loss += loss.item()

        avg_val = val_loss / max(len(val_dl), 1)
        print(
            f"Epoch {epoch:03d} | "
            f"train {avg_train:.4f} (mse {avg_mse:.4f} tcr {avg_tcr:.4f}) | "
            f"val {avg_val:.4f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

        # ── Checkpointing ─────────────────────────────────────────────────
        ckpt = {
            "epoch":          epoch,
            "model":          model.state_dict(),
            "bbox_enc":       bbox_enc.state_dict(),
            "null_bbox":      null_bbox.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "best_val_loss":  best_val_loss,
        }
        torch.save(ckpt, ckpt_path)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_path = os.path.join(args.weights_dir, f"ra_synthgen_{C.MODEL_TAG}_best.pth")
            torch.save(ckpt, best_path)
            print(f"  ↳ New best model saved  (val {avg_val:.4f})")

        # ── Visualisation ─────────────────────────────────────────────────
        if (epoch + 1) % C.VIZ_EVERY_N_EPOCHS == 0:
            val_batch = next(iter(val_dl))
            viz_batch(epoch, val_batch, model, dino_enc, bbox_enc, C.DEVICE, args.viz_dir,
                      radial_root=args.radial_root,
                      use_cached_dino=C.USE_CACHED_DINO)
            model.train(); dino_enc.train(); bbox_enc.train()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Weights saved in {args.weights_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RA-SynthGen training")
    p.add_argument("--radial-root",  default=C.RADIAL_ROOT)
    p.add_argument("--labels-dir",   default=C.LABELS_DIR,
                   help="EchoFusion labels_x/ directory (primary)")
    p.add_argument("--labels-csv",   default=C.LABELS_CSV,
                   help="Fallback: radial_bbox_labels.csv (leave None to use --labels-dir)")
    p.add_argument("--psf-path",     default=C.PSF_PATH)
    p.add_argument("--weights-dir",  default=C.WEIGHTS_DIR)
    p.add_argument("--viz-dir",      default=C.VIZ_DIR)
    p.add_argument("--epochs",       type=int,   default=C.EPOCHS)
    p.add_argument("--batch-size",   type=int,   default=C.BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=C.LR)
    p.add_argument("--lambda-tcr",   type=float, default=C.LAMBDA_TCR)
    args = p.parse_args()
    train(args)
