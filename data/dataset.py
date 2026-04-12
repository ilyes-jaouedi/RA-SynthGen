"""
RADIalSynthDataset — dataloader for RA-SynthGen.

Each sample returns:
  image     : (3, 266, 518)       DINOv2-normalised camera frame
              OR None when cached DINOv2 features are used (dino_feat is returned instead)
  dino_feat : (703, 1024) float16 pre-computed DINOv2 features (when use_cached_dino=True)
  x1        : (1, 256, 256)       GT Bartlett RA map, log-normalised to [-1, 1]
  x0        : (1, 256, 256)       PSF-blob prior from 3D bbox labels
  bboxes    : (MAX_BBOXES, 7)     3D boxes [x,y,z,dim_x,dim_y,dim_z,theta], zero-padded
  bbox_mask : (MAX_BBOXES,)       bool — True for real boxes
  frame_id  : int                 echo_frame_id

--- Data layout (produced by data/extract_from_records.py) ---

  radial_root/
    index.csv          ← echo_frame_id, image, radar, seq_name, local_frame_idx
    camera/
      000018.jpg        ← camera frames keyed by echo_frame_id
    radar_FFT/
      000018.npy        ← Bartlett RA maps (512, 751, 11)

Labels (EchoFusion format, one .txt per frame):
  labels_dir/
    000018.txt          ← one box per line: x y z dim_x dim_y dim_z theta class

--- Backward-compatible mode (old RADIal CSV labels) ---

Pass labels_csv instead of labels_dir.  The CSV columns must include:
  numSample (or frame_id), radar_X_m, radar_Y_m, radar_Z_m,
  dim_X_m, dim_Y_m, dim_Z_m, rotation_y
"""

import os
import glob
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from models.psf_prior import build_psf_x0, load_psf_kernel

# ── Constants ──────────────────────────────────────────────────────────────────
RADAR_SIZE  = 256
AZ_MIN_DEG  = -75.0
AZ_MAX_DEG  =  75.0
R_MAX_M     = 103.0
RAW_AZ_BINS = 751
RAW_R_BINS  = 512
ELEV_IDX    = 5           # road-plane elevation slice
MAX_BBOXES  = 8

LOG_MIN = 15.0
LOG_MAX = 23.0

# DINOv2 / ImageNet normalisation (also matches ViT-L/14 used by DINOv2)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


# ── Normalisation helpers ──────────────────────────────────────────────────────

def normalise_ra(ra: np.ndarray) -> np.ndarray:
    """log1p → clip [LOG_MIN, LOG_MAX] → scale to [-1, 1]."""
    ra = np.log1p(ra.astype(np.float32))
    ra = np.clip(ra, LOG_MIN, LOG_MAX)
    ra = (ra - LOG_MIN) / (LOG_MAX - LOG_MIN)
    return 2.0 * ra - 1.0


def denormalise_ra(ra: np.ndarray) -> np.ndarray:
    """Inverse of normalise_ra."""
    ra = (ra + 1.0) / 2.0
    ra = ra * (LOG_MAX - LOG_MIN) + LOG_MIN
    return np.expm1(ra)


# ── Dataset ───────────────────────────────────────────────────────────────────

class RADIalSynthDataset(Dataset):
    """
    Parameters
    ----------
    radial_root : str
        Root folder produced by extract_from_records.py.
        Must contain index.csv, camera/, and radar_FFT/.
    psf_path : str
        Path to radial_psf_analytic.npy.
    labels_dir : str | None
        Directory of EchoFusion .txt label files, one per frame.
        Format per line: x y z dim_x dim_y dim_z theta [class]
        Files named {echo_frame_id:06d}.txt.
        If None, labels_csv must be provided.
    labels_csv : str | None
        Fallback: path to radial_bbox_labels.csv (old format).
        Used only when labels_dir is None.
    split : str
        'train' | 'val' | 'all'
    val_fraction : float
        Fraction reserved for validation (chronological tail of index.csv).
    max_bboxes : int
        Padding/truncation size for bbox tensors.
    """

    def __init__(
        self,
        radial_root:      str,
        psf_path:         str,
        labels_dir:       str | None = None,
        labels_csv:       str | None = None,
        split:            str = "train",
        val_fraction:     float = 0.2,
        max_bboxes:       int = MAX_BBOXES,
        use_cached_dino:  bool = False,
    ):
        if labels_dir is None and labels_csv is None:
            raise ValueError("Provide either labels_dir (EchoFusion .txt) or labels_csv.")

        self.radial_root     = radial_root
        self.labels_dir      = labels_dir
        self.max_bboxes      = max_bboxes
        self.split           = split
        self.use_cached_dino = use_cached_dino

        # ── DINOv2 feature cache directory ─────────────────────────────────────
        self._dino_dir = os.path.join(radial_root, "dino_features")
        if use_cached_dino and not os.path.isdir(self._dino_dir):
            raise FileNotFoundError(
                f"DINOv2 cache not found at {self._dino_dir}. "
                "Run:  python data/cache_dino_features.py"
            )

        # ── Image transform (used when NOT using cached features) ──────────────
        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((266, 518)),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ])

        # ── PSF kernel ─────────────────────────────────────────────────────────
        self.psf_kernel = load_psf_kernel(psf_path)

        # ── Load frame index ───────────────────────────────────────────────────
        index_path = os.path.join(radial_root, "index.csv")
        if os.path.exists(index_path):
            df = pd.read_csv(index_path)
            # Columns: echo_frame_id, image, radar, seq_name, local_frame_idx
            self._frames = self._build_from_index(df)
        else:
            # Fall back to scanning camera/ and radar_FFT/ for matching files
            self._frames = self._build_from_dirs()

        # ── Backward-compat: CSV labels ────────────────────────────────────────
        if labels_dir is None and labels_csv is not None:
            df_labels = pd.read_csv(labels_csv)
            df_labels.columns = [c.strip().lower() for c in df_labels.columns]
            df_labels = df_labels.rename(columns={
                "numsample":  "frame_id",
                "radar_x_m":  "x",
                "radar_y_m":  "y",
                "radar_z_m":  "z",
                "dim_x_m":    "dim_x",
                "dim_y_m":    "dim_y",
                "dim_z_m":    "dim_z",
                "rotation_y": "theta",
            })
            self._csv_labels = df_labels.groupby("frame_id")
        else:
            self._csv_labels = None

        # ── Train / val split (chronological) ─────────────────────────────────
        n_val = max(1, int(len(self._frames) * val_fraction))
        if split == "train":
            self._frames = self._frames[:-n_val]
        elif split == "val":
            self._frames = self._frames[-n_val:]
        # else 'all' — no cut

        print(
            f"RADIalSynthDataset [{split}]: {len(self._frames)} frames"
        )

    # ── Frame list construction ────────────────────────────────────────────────

    def _build_from_index(self, df: pd.DataFrame):
        """
        Build list of (echo_frame_id, img_abs_path, npy_abs_path) from index.csv.
        Relative paths in the CSV are resolved against radial_root.
        """
        frames = []
        cam_dir_fallback   = self._find_cam_dir()
        radar_dir_fallback = self._find_radar_dir()

        for _, row in df.iterrows():
            fid = int(row["echo_frame_id"])

            # Try path from index first, then fallback to scanned dirs
            img_path = os.path.join(self.radial_root, str(row.get("image", "")))
            npy_path = os.path.join(self.radial_root, str(row.get("radar",  "")))

            if not os.path.exists(img_path):
                img_path = self._find_img(cam_dir_fallback,   fid)
            if not os.path.exists(npy_path):
                npy_path = self._find_npy(radar_dir_fallback, fid)

            if img_path and npy_path and os.path.exists(img_path) and os.path.exists(npy_path):
                frames.append((fid, img_path, npy_path))

        return frames

    def _build_from_dirs(self):
        """Scan camera/ and radar_FFT/ for matching {frame_id:06d}.* files."""
        cam_dir   = self._find_cam_dir()
        radar_dir = self._find_radar_dir()

        npy_ids = {
            int(os.path.basename(p).split(".")[0])
            for p in glob.glob(os.path.join(radar_dir, "*.npy"))
            if os.path.basename(p).split(".")[0].isdigit()
        }

        frames = []
        for fid in sorted(npy_ids):
            img_path = self._find_img(cam_dir,   fid)
            npy_path = self._find_npy(radar_dir, fid)
            if img_path and npy_path:
                frames.append((fid, img_path, npy_path))
        return frames

    def _find_cam_dir(self):
        for name in ("camera", "img", "images"):
            p = os.path.join(self.radial_root, name)
            if os.path.isdir(p):
                return p
        raise FileNotFoundError(
            f"No camera folder under {self.radial_root}. "
            "Expected camera/, img/, or images/."
        )

    def _find_radar_dir(self):
        for name in ("radar_FFT", "fft_data", "FFT_data", "radar_maps", "Spectral"):
            p = os.path.join(self.radial_root, name)
            if os.path.isdir(p):
                return p
        raise FileNotFoundError(
            f"No radar folder under {self.radial_root}. "
            "Expected radar_FFT/, fft_data/, or radar_maps/."
        )

    @staticmethod
    def _find_img(cam_dir: str, fid: int):
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(cam_dir, f"{fid:06d}{ext}")
            if os.path.exists(p):
                return p
        return None

    @staticmethod
    def _find_npy(radar_dir: str, fid: int):
        p = os.path.join(radar_dir, f"{fid:06d}.npy")
        return p if os.path.exists(p) else None

    # ── Label loading ─────────────────────────────────────────────────────────

    def _get_bboxes(self, frame_id: int):
        """
        Returns (MAX_BBOXES, 7) padded float tensor and (MAX_BBOXES,) bool mask.
        Row format: [x, y, z, dim_x, dim_y, dim_z, theta]  in radar frame.
        """
        boxes = np.zeros((self.max_bboxes, 7), dtype=np.float32)
        mask  = np.zeros(self.max_bboxes,      dtype=bool)

        if self.labels_dir is not None:
            # EchoFusion txt format: x y z dim_x dim_y dim_z theta [class]
            txt_path = os.path.join(self.labels_dir, f"{frame_id:06d}.txt")
            if os.path.exists(txt_path):
                lines = [l.strip().split() for l in open(txt_path) if l.strip()]
                for i, parts in enumerate(lines[:self.max_bboxes]):
                    vals = [float(v) for v in parts[:7]]
                    boxes[i, :len(vals)] = vals
                    mask[i] = True
        elif self._csv_labels is not None:
            if frame_id in self._csv_labels.groups:
                rows = self._csv_labels.get_group(frame_id)
                cols = [c for c in ["x", "y", "z", "dim_x", "dim_y", "dim_z", "theta"]
                        if c in rows.columns]
                arr  = rows[cols].values.astype(np.float32)
                n    = min(len(arr), self.max_bboxes)
                boxes[:n, :len(cols)] = arr[:n]
                mask[:n]              = True

        return torch.from_numpy(boxes), torch.from_numpy(mask)

    # ── RA map loading ────────────────────────────────────────────────────────

    @staticmethod
    def _load_fft(path: str) -> np.ndarray:
        """
        Load a Bartlett RA map .npy file.

        Supported shapes (produced by RadarProcessor.process_frame):
          (512, 751, 11)  ← standard output from extract_from_records.py (range, az, elev)
          (512, 751)      ← pre-projected 2D map
          (11, 512, 751)  ← elevation-first variant
          4D              ← ADC-domain, simplified magnitude fallback
        """
        data = np.load(path, allow_pickle=False)

        if data.ndim == 2:
            return data.astype(np.float32)

        if data.ndim == 3:
            if data.shape[2] == 11:
                # (range=512, az=751, elev=11) — standard output
                return np.abs(data[:, :, ELEV_IDX]).astype(np.float32)
            if data.shape[0] == 11:
                # (elev=11, range, az)
                return np.abs(data[ELEV_IDX]).astype(np.float32)
            # Generic 3D: assume last dim is auxiliary
            return np.abs(data[:, :, 0]).astype(np.float32)

        if data.ndim == 4:
            return np.abs(data).mean(axis=(0, -1)).astype(np.float32)

        raise ValueError(f"Unexpected shape {data.shape} in {path}")

    def _process_ra(self, raw: np.ndarray) -> torch.Tensor:
        """(H, W) power map → (1, 256, 256) normalised tensor."""
        raw = raw[:RAW_R_BINS, :RAW_AZ_BINS]
        ra  = cv2.resize(raw, (RADAR_SIZE, RADAR_SIZE), interpolation=cv2.INTER_LINEAR)
        ra  = normalise_ra(ra)
        return torch.from_numpy(ra).float().unsqueeze(0)

    # ── DataLoader interface ──────────────────────────────────────────────────

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        frame_id, img_path, fft_path = self._frames[idx]

        # ── Image / DINOv2 features ────────────────────────────────────────────
        if self.use_cached_dino:
            feat_path = os.path.join(self._dino_dir, f"{frame_id:06d}.npy")
            dino_feat = torch.from_numpy(
                np.load(feat_path).astype(np.float32)
            )                                           # (703, 1024)
            image = None                                # not needed downstream
        else:
            img_bgr   = cv2.imread(img_path)
            img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            image     = self.img_transform(img_rgb)     # (3, 266, 518)
            dino_feat = None

        # Ground-truth RA map (x1)
        raw_ra  = self._load_fft(fft_path)
        x1      = self._process_ra(raw_ra)              # (1, 256, 256)

        # Bbox labels
        bboxes, bbox_mask = self._get_bboxes(frame_id)  # (MAX_BBOXES,7), (MAX_BBOXES,)

        # PSF x0 prior (use only the real — unpadded — boxes)
        real_boxes = bboxes[bbox_mask].numpy()           # (N, 7)
        x0 = build_psf_x0(
            real_boxes,
            self.psf_kernel,
            range_bins=RADAR_SIZE,
            az_bins=RADAR_SIZE,
            az_min_deg=AZ_MIN_DEG,
            az_max_deg=AZ_MAX_DEG,
            r_max_m=R_MAX_M,
        )                                               # (1, 256, 256)

        sample = {
            "x1":        x1,
            "x0":        x0,
            "bboxes":    bboxes,
            "bbox_mask": bbox_mask,
            "frame_id":  frame_id,
        }
        if self.use_cached_dino:
            sample["dino_feat"] = dino_feat
        else:
            sample["image"] = image
        return sample


# ── Sanity check (run directly) ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Verify RADIalSynthDataset loads correctly")
    p.add_argument(
        "--root",
        default=os.path.join(_HERE, "radial"),
        help="radial_root produced by extract_from_records.py",
    )
    p.add_argument(
        "--labels-dir",
        default=r"C:\Users\Ilyes\Desktop\On-RADIal\EchoFusion_data\labels_x",
        help="Directory with EchoFusion {frame_id:06d}.txt label files",
    )
    p.add_argument(
        "--labels-csv",
        default=None,
        help="Fallback: path to radial_bbox_labels.csv",
    )
    p.add_argument(
        "--psf",
        default=os.path.join(_HERE, "..", "assets", "radial_psf_analytic.npy"),
    )
    args = p.parse_args()

    ds = RADIalSynthDataset(
        radial_root=args.root,
        psf_path=args.psf,
        labels_dir=args.labels_dir if args.labels_dir else None,
        labels_csv=args.labels_csv if args.labels_csv else None,
        split="all",
    )

    if len(ds) == 0:
        print("Dataset is EMPTY — check paths above.")
    else:
        s = ds[0]
        print(f"image    : {s['image'].shape}")
        print(f"x1       : {s['x1'].shape}  range [{s['x1'].min():.3f}, {s['x1'].max():.3f}]")
        print(f"x0       : {s['x0'].shape}  range [{s['x0'].min():.3f}, {s['x0'].max():.3f}]")
        print(f"bboxes   : {s['bboxes'].shape}  real={s['bbox_mask'].sum().item()}")
        print(f"frame_id : {s['frame_id']}")
        print("OK — dataset is loading correctly.")
