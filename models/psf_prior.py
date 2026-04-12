"""
PSF-blob x0 prior constructor.

For each 3D bounding box (x, y, z, dim_x, dim_y, dim_z, theta) in the
radar frame (X=forward, Y=left, Z=up), we:

  1. Compute (range, azimuth) of the box centre.
  2. Convert to (range_bin, az_bin) in the 256x256 RA grid.
  3. Scale the calibrated PSF by a proxy RCS amplitude (dim_x x dim_y).
  4. Stamp the PSF onto the canvas at that location.

The PSF kernel is the REAL calibrated PSF — the Gram column at boresight
computed from CalibrationTable.npy, not a heuristic Gaussian:

    K(theta, phi) = |G_w^H  g_boresight|^2     (az x el)

    where  G_w = diag(window) @ G_calib         (192 x 751*11)
           g_boresight = G_w[:, az0*11 + el0]   steering vec at (0,0)

The range PSF is the actual |FFT(Hamming(512))|^2 mainlobe (not a Gaussian).

Two entry points:
  load_psf_kernel(path)          -- load pre-saved (r, az) 2D kernel .npy
  compute_psf_from_calib(path)   -- compute from CalibrationTable.npy live
"""

import os
import numpy as np
import torch


# ── Real calibrated PSF ───────────────────────────────────────────────────────

def compute_psf_from_calib(
    calib_path: str,
    target_az_bins: int = 64,
    target_r_bins: int = 32,
    num_range_samples: int = 512,
) -> np.ndarray:
    """
    Compute the true 2D (range, azimuth) PSF kernel from CalibrationTable.npy.

    Azimuth PSF  -- Gram column at boresight (road-plane elevation):
        G_w      = diag(antenna_window) @ G_calib     (192, 751*11)
        g_0      = G_w[:, boresight_flat_idx]          (192,)
        K_az_el  = |G_w^H @ g_0|^2                    (751*11,) -> (751, 11)
        psf_az   = K_az_el[:, el0_idx]                (751,)  -- el slice at 0 deg

    Range PSF  -- |fftshift(FFT(Hamming(512)))|^2, central lobe extracted:
        psf_r    = mainlobe of Hamming-windowed range FFT  (target_r_bins,)

    Both are normalised to max = 1 then combined via outer product.

    Parameters
    ----------
    calib_path : str
        Path to CalibrationTable.npy (the RADIal SignalProcessing file).
    target_az_bins : int
        Width of returned kernel (azimuth dimension).
    target_r_bins : int
        Height of returned kernel (range dimension).
    num_range_samples : int
        Number of range samples used during beamforming (512 for RADIal).

    Returns
    -------
    (target_r_bins, target_az_bins) float32, max = 1.
    """
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"CalibrationTable not found: {calib_path}")

    aoa_data  = np.load(calib_path, allow_pickle=True).item()
    raw_calib = aoa_data["Signal"]          # (751, 192, 11) complex128
    window    = aoa_data["H"][0]            # (192,) antenna tapering
    az_table  = aoa_data["Azimuth_table"]   # (751,) degrees
    el_table  = aoa_data["Elevation_table"] # (11,)  degrees

    num_az = raw_calib.shape[0]   # 751
    num_el = raw_calib.shape[2]   # 11

    # ── Build windowed steering matrix G_w: (192, num_az*num_el) ─────────────
    # raw_calib is (az, ant, el) -> transpose to (ant, az, el) -> reshape
    G_calib = np.transpose(raw_calib, (1, 0, 2)).reshape(192, -1)  # (192, 751*11)
    G_w     = window[:, np.newaxis] * G_calib                       # (192, 751*11)

    # ── Boresight index (az~0, el~0) ─────────────────────────────────────────
    az0_idx = int(np.argmin(np.abs(az_table)))
    el0_idx = int(np.argmin(np.abs(el_table)))
    boresight_flat = az0_idx * num_el + el0_idx

    g_boresight = G_w[:, boresight_flat]                            # (192,)

    # ── Gram column: K(theta,phi) = |G_w^H @ g_0|^2 ──────────────────────────
    gram_col = np.abs(G_w.conj().T @ g_boresight) ** 2             # (751*11,)
    K_az_el  = gram_col.reshape(num_az, num_el)                     # (751, 11)

    # ── Azimuth PSF at road-plane elevation ───────────────────────────────────
    psf_az_raw = K_az_el[:, el0_idx].astype(np.float64)            # (751,)
    psf_az_raw = psf_az_raw / psf_az_raw.max()

    # Resize 751 -> target_az_bins
    from scipy.ndimage import zoom as _zoom
    psf_az = _zoom(psf_az_raw, target_az_bins / num_az, order=1).astype(np.float32)
    psf_az = psf_az / psf_az.max()

    # ── Range PSF: |fftshift(FFT(Hamming))|^2, central lobe ──────────────────
    w_range      = np.hamming(num_range_samples)
    range_fft    = np.fft.fftshift(np.fft.fft(w_range, n=num_range_samples))
    psf_range_full = np.abs(range_fft) ** 2                        # (512,)
    psf_range_full = psf_range_full / psf_range_full.max()

    # Extract the central mainlobe (target_r_bins samples around the peak)
    centre = num_range_samples // 2
    half   = target_r_bins // 2
    psf_r  = psf_range_full[centre - half : centre + half].astype(np.float32)
    # Pad or trim if necessary (edge case when target_r_bins is odd)
    if len(psf_r) < target_r_bins:
        psf_r = np.pad(psf_r, (0, target_r_bins - len(psf_r)))
    psf_r = psf_r[:target_r_bins]
    psf_r = psf_r / psf_r.max()

    # ── 2D separable kernel via outer product ─────────────────────────────────
    psf_2d = np.outer(psf_r, psf_az).astype(np.float32)            # (r_bins, az_bins)
    psf_2d = psf_2d / psf_2d.max()

    return psf_2d


def save_psf_kernel(psf_2d: np.ndarray, out_path: str):
    """Cache the computed PSF kernel so it doesn't need to be recomputed."""
    np.save(out_path, psf_2d)
    print(f"PSF kernel saved -> {out_path}  shape={psf_2d.shape}")


# ── Load pre-saved 2D kernel ──────────────────────────────────────────────────

def load_psf_kernel(
    path: str,
    target_az_bins: int = 64,
    target_r_bins: int = 32,
) -> np.ndarray:
    """
    Load a pre-saved (range, azimuth) PSF kernel .npy file.

    The file must contain a 2D float array of shape (H, W).
    If the shape doesn't match (target_r_bins, target_az_bins) it is resized.

    To compute the kernel from CalibrationTable.npy for the first time use:
        psf = compute_psf_from_calib(calib_path)
        save_psf_kernel(psf, "assets/radial_psf_calibrated.npy")

    Returns
    -------
    (target_r_bins, target_az_bins) float32, max = 1.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PSF kernel not found at {path}.\n"
            "Generate it first with:\n"
            "  python models/psf_prior.py --calib <CalibrationTable.npy> "
            "--save assets/radial_psf_calibrated.npy"
        )

    raw = np.load(path, allow_pickle=False)

    if raw.ndim != 2:
        raise ValueError(
            f"Expected a 2D (range, az) PSF kernel in {path}, got shape {raw.shape}.\n"
            "Use compute_psf_from_calib() to build the correct kernel."
        )

    if np.iscomplexobj(raw):
        raw = np.abs(raw)

    raw = raw.astype(np.float32)

    # Resize if needed
    if raw.shape != (target_r_bins, target_az_bins):
        import cv2 as _cv2
        raw = _cv2.resize(raw, (target_az_bins, target_r_bins), interpolation=_cv2.INTER_LINEAR)

    raw = raw / max(raw.max(), 1e-9)
    return raw


# ── Coordinate conversions ────────────────────────────────────────────────────

def range_to_bin(r_m: float, r_max_m: float, n_bins: int) -> int:
    """Continuous range [0, r_max_m] -> discrete bin [0, n_bins-1]."""
    return int(np.clip(r_m / r_max_m * n_bins, 0, n_bins - 1))


def azimuth_to_bin(az_deg: float, az_min_deg: float, az_max_deg: float, n_bins: int) -> int:
    """Azimuth angle -> discrete bin [0, n_bins-1]."""
    frac = (az_deg - az_min_deg) / (az_max_deg - az_min_deg)
    return int(np.clip(frac * n_bins, 0, n_bins - 1))


# ── PSF stamping ──────────────────────────────────────────────────────────────

def _stamp_psf(
    canvas: np.ndarray,
    psf: np.ndarray,
    r_bin: int,
    az_bin: int,
    amplitude: float,
) -> None:
    """Add amplitude * psf centred at (r_bin, az_bin) onto canvas in-place."""
    H, W = canvas.shape
    h, w = psf.shape

    r0, c0 = r_bin - h // 2, az_bin - w // 2
    r1, c1 = r0 + h, c0 + w

    pr0 = max(0, -r0);  pc0 = max(0, -c0)
    pr1 = h - max(0, r1 - H);  pc1 = w - max(0, c1 - W)

    cr0 = max(0, r0);  cc0 = max(0, c0)
    cr1 = min(H, r1);  cc1 = min(W, c1)

    if pr1 > pr0 and pc1 > pc0:
        canvas[cr0:cr1, cc0:cc1] += amplitude * psf[pr0:pr1, pc0:pc1]


# ── Main API ──────────────────────────────────────────────────────────────────

def build_psf_x0(
    bboxes: np.ndarray,
    psf_kernel: np.ndarray,
    range_bins: int = 256,
    az_bins: int = 256,
    az_min_deg: float = -75.0,
    az_max_deg: float = 75.0,
    r_max_m: float = 103.0,
    rcs_mode: str = "area",   # 'area' | 'uniform'
) -> torch.Tensor:
    """
    Build a PSF-blob x0 prior from 3D bounding boxes.

    Parameters
    ----------
    bboxes : (N, 7) array  [x, y, z, dim_x, dim_y, dim_z, theta]  radar frame.
    psf_kernel : (H_psf, W_psf) normalised calibrated PSF kernel.
    rcs_mode : 'area'    -> amplitude = (dim_x * dim_y) / 9.4  (car RCS proxy)
               'uniform' -> amplitude = 1

    Returns
    -------
    (1, range_bins, az_bins) torch.FloatTensor in [-1, 1]
    """
    canvas = np.zeros((range_bins, az_bins), dtype=np.float32)

    if bboxes is not None and len(bboxes) > 0:
        for box in bboxes:
            x_b, y_b, z_b, dim_x, dim_y, dim_z, theta = (
                float(box[0]), float(box[1]), float(box[2]),
                float(box[3]), float(box[4]), float(box[5]),
                float(box[6]),
            )

            r_m    = float(np.sqrt(x_b ** 2 + y_b ** 2))
            az_deg = float(np.degrees(np.arctan2(y_b, x_b)))

            if r_m <= 0 or r_m > r_max_m:
                continue
            if az_deg < az_min_deg or az_deg > az_max_deg:
                continue

            r_bin  = range_to_bin(r_m, r_max_m, range_bins)
            az_bin = azimuth_to_bin(az_deg, az_min_deg, az_max_deg, az_bins)

            if rcs_mode == "area":
                amplitude = float(np.clip(dim_x * dim_y, 0.1, 50.0)) / 9.4
            else:
                amplitude = 1.0

            _stamp_psf(canvas, psf_kernel, r_bin, az_bin, amplitude)

    LOG_MIN, LOG_MAX = 15.0, 23.0
    if canvas.max() > 0:
        raw = canvas * (np.exp(LOG_MAX) - 1)
        raw = np.log1p(raw)
        raw = np.clip(raw, LOG_MIN, LOG_MAX)
        raw = (raw - LOG_MIN) / (LOG_MAX - LOG_MIN)
        out = 2.0 * raw - 1.0
    else:
        out = canvas

    return torch.from_numpy(out).float().unsqueeze(0)   # (1, range_bins, az_bins)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(
        description="Compute / test the calibrated PSF kernel"
    )
    p.add_argument(
        "--calib",
        default=r"C:\Users\Ilyes\Desktop\On-RADIal\RADIal\SignalProcessing\CalibrationTable.npy",
        help="Path to CalibrationTable.npy",
    )
    p.add_argument(
        "--save",
        default=None,
        help="Where to save the computed 2D kernel (e.g. assets/radial_psf_calibrated.npy)",
    )
    p.add_argument(
        "--load",
        default=None,
        help="Load a pre-saved kernel instead of computing (skips --calib)",
    )
    p.add_argument(
        "--out-img",
        default="psf_calibrated_test.png",
        help="Output PNG path",
    )
    args = p.parse_args()

    if args.load:
        print(f"Loading pre-saved PSF from {args.load}")
        psf = load_psf_kernel(args.load)
    else:
        print(f"Computing calibrated PSF from {args.calib} ...")
        psf = compute_psf_from_calib(args.calib)
        print(f"  Kernel shape: {psf.shape}  max={psf.max():.4f}")
        if args.save:
            save_psf_kernel(psf, args.save)

    # ── Visual test: 3 synthetic cars ─────────────────────────────────────────
    test_boxes = np.array([
        [ 15.0,  0.0,  0.5,  4.7, 2.0, 1.7,  0.0],
        [ 30.0,  8.0,  0.5,  4.7, 2.0, 1.7,  0.1],
        [ 50.0, -5.0,  0.5,  4.7, 2.0, 1.7, -0.1],
    ], dtype=np.float32)
    x0 = build_psf_x0(test_boxes, psf)
    print(f"x0 shape={x0.shape}  range=[{x0.min():.3f}, {x0.max():.3f}]")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: 2D PSF kernel
    axes[0].imshow(psf, origin="lower", cmap="hot", aspect="auto")
    axes[0].set_title(f"Calibrated PSF kernel  {psf.shape}", fontsize=11)
    axes[0].set_xlabel("Azimuth bins"); axes[0].set_ylabel("Range bins")

    # Panel 2: PSF slices
    cx = psf.shape[1] // 2
    axes[1].plot(psf[psf.shape[0]//2, :], label="Azimuth slice (at peak range)")
    axes[1].plot(psf[:, cx],              label="Range slice  (at peak az)", ls="--")
    axes[1].set_title("PSF slices through mainlobe peak", fontsize=11)
    axes[1].set_xlabel("Bin"); axes[1].set_ylabel("Normalised amplitude")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Panel 3: x0 prior for 3 synthetic cars
    im = axes[2].imshow(x0[0].numpy(), origin="lower", cmap="inferno",
                        vmin=-1, vmax=1, aspect="auto")
    axes[2].set_title("x0 prior  (3 synthetic targets)", fontsize=11)
    axes[2].set_xlabel("Azimuth bin"); axes[2].set_ylabel("Range bin")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("Calibrated PSF from CalibrationTable.npy", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.out_img, dpi=150, bbox_inches="tight")
    print(f"Saved -> {args.out_img}")
