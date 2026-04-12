"""
Extract camera images and Bartlett RA maps for all EchoFusion test frames.

Reads eval_index.csv (echo_frame_id -> seq_name + local_frame_idx) and
eval_split.csv (echo_frame_id -> split), then for each test-split frame:
  1. Opens the RECORD@ sequence with SyncReader
  2. Extracts the camera frame (BGR -> saved as JPEG)
  3. Runs PyTorch Bartlett beamforming -> (512, 751, 11) RA map
  4. Saves as  data/radial/camera/{echo_frame_id:06d}.jpg
              data/radial/radar_FFT/{echo_frame_id:06d}.npy

Writes data/radial/index.csv with columns:
    echo_frame_id, image, radar, seq_name, local_frame_idx

Usage (from RA-SynthGen root):
    python data/extract_from_records.py \
        --radial-root C:/Users/Ilyes/Desktop/On-RADIal/RADIal \
        --eval-index C:/Users/Ilyes/Desktop/On-RADIal/EchoFusion_data/eval_index.csv \
        --eval-split  C:/Users/Ilyes/Desktop/On-RADIal/EchoFusion_data/eval_split.csv \
        --out-dir     data/radial \
        [--split test]           # default: test
        [--dry-run]              # print frame list but don't extract
        [--skip-existing]        # skip frames whose .npy already exists
        [--cpu]                  # force CPU beamforming (slower, no GPU needed)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm
from scipy.ndimage import zoom

# ── Path setup ────────────────────────────────────────────────────────────────
# Locate the On-RADIal root from this script's location or the --radial-root arg
_HERE = os.path.dirname(os.path.abspath(__file__))
_RA_SYNTHGEN_ROOT = os.path.dirname(_HERE)


def _add_dbreader_to_path(radial_root: str):
    """Add DBReader to sys.path so SyncReader can be imported."""
    candidates = [
        os.path.join(radial_root, "DBReader"),
        os.path.join(os.path.dirname(radial_root), "RADIal", "DBReader"),
    ]
    for c in candidates:
        if os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)
    # Also add the parent of radial_root so `from DBReader.DBReader import ...` works
    parent = os.path.dirname(radial_root)
    if parent not in sys.path:
        sys.path.insert(0, parent)


# ── RadarProcessor (ported from create_dense_dataset.py) ─────────────────────

class RadarProcessor:
    """
    PyTorch-accelerated Bartlett beamformer.
    Produces (Range=512, Azimuth=751, Elevation=11) float32 maps.

    Ported verbatim from On-RADIal/global_scripts/data_prep/create_dense_dataset.py.
    """

    def __init__(self, calib_path: str, device: str = "cuda"):
        dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.device = torch.device(dev)
        print(f"RadarProcessor -> {self.device}")

        # Radar parameters (RADIal)
        self.numSamplePerChirp  = 512
        self.numRxPerChip       = 4
        self.numChirps          = 256
        self.numRxAnt           = 16
        self.numTxAnt           = 12
        self.numReducedDoppler  = 16
        self.numChirpsPerLoop   = 16

        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"CalibrationTable not found: {calib_path}")

        aoa_data  = np.load(calib_path, allow_pickle=True).item()
        raw_calib = aoa_data["Signal"]                # (751, 192, 11) complex128

        self.calib_mat_np = raw_calib
        self.num_az = self.calib_mat_np.shape[0]      # 751
        self.num_el = self.calib_mat_np.shape[2]      # 11

        # Reshape -> (192, Az*El) for batch matmul
        calib_reshaped = np.transpose(self.calib_mat_np, (1, 0, 2)).reshape(192, -1)
        self.calib_mat = torch.from_numpy(calib_reshaped).to(self.device).cfloat()

        # Windows
        w_range   = np.hamming(self.numSamplePerChirp).astype(np.float32)
        w_doppler = np.hamming(self.numChirps).astype(np.float32)
        w_angle   = aoa_data["H"][0].astype(np.float32)

        self.window_range   = torch.from_numpy(w_range).to(self.device)
        self.window_doppler = torch.from_numpy(w_doppler).to(self.device)
        self.window_angle   = torch.from_numpy(w_angle).to(self.device)

        # Doppler index table for MIMO virtual array construction
        self.dividend_constant_arr = np.arange(
            0, self.numReducedDoppler * self.numChirpsPerLoop, self.numReducedDoppler
        )
        doppler_indices_np = np.zeros((self.numChirps, 12), dtype=np.int64)
        for d in range(self.numChirps):
            shifts         = self.dividend_constant_arr
            seq            = (d + shifts) % self.numChirps
            seq_selected   = np.concatenate(([seq[0]], seq[5:]))
            doppler_indices_np[d] = seq_selected
        self.doppler_indices = torch.from_numpy(doppler_indices_np).to(self.device)

    # ------------------------------------------------------------------
    def get_mimo_spectrum(self, adc_data: dict) -> torch.Tensor:
        """Returns (Range=512, Doppler=256, Antennas=192) complex tensor."""
        chips = []
        for i in range(4):
            raw  = adc_data[f"radar_ch{i}"]["data"]
            comp = raw[0::2] + 1j * raw[1::2]
            comp = comp.reshape(
                (self.numSamplePerChirp, self.numRxPerChip, self.numChirps),
                order="F",
            )
            comp = comp.transpose((0, 2, 1))          # (512, 256, 4)
            chips.append(torch.from_numpy(comp))

        frame = torch.cat([chips[3], chips[0], chips[1], chips[2]], dim=2)  # (512,256,16)
        frame = frame.to(self.device).cfloat()
        frame = frame - torch.mean(frame, dim=(0, 1), keepdim=True)

        # Range FFT
        range_fft = torch.fft.fft(frame * self.window_range.view(-1, 1, 1), dim=0)
        # Doppler FFT
        dop_fft   = torch.fft.fft(range_fft * self.window_doppler.view(1, -1, 1), dim=1)

        # MIMO virtual array construction
        dop_fft_perm = dop_fft.permute(1, 0, 2)      # (256, 512, 16)
        mimo_list = []
        for tx in range(12):
            idxs = self.doppler_indices[:, tx]
            mimo_list.append(dop_fft_perm[idxs, :, :])
        mimo_stack    = torch.stack(mimo_list, dim=2)  # (256, 512, 12, 16)
        mimo_spectrum = mimo_stack.reshape(256, 512, 192).permute(1, 0, 2)  # (512,256,192)
        mimo_spectrum = mimo_spectrum * self.window_angle.view(1, 1, -1)

        return mimo_spectrum

    # ------------------------------------------------------------------
    def process_frame(self, adc_data: dict) -> np.ndarray:
        """
        Returns (512, 751, 11) float32 Bartlett RA map.
        """
        mimo_spectrum = self.get_mimo_spectrum(adc_data)

        num_range   = 512
        num_targets = self.calib_mat.shape[1]              # 751 * 11 = 8261
        chunk_size  = 32
        radar_3d_cpu = np.zeros((num_range, num_targets), dtype=np.float32)

        for r_start in range(0, num_range, chunk_size):
            r_end    = min(r_start + chunk_size, num_range)
            sig_chunk = mimo_spectrum[r_start:r_end]       # (chunk, 256, 192)
            bf_chunk  = torch.matmul(sig_chunk, self.calib_mat)
            mag_chunk = torch.abs(bf_chunk)
            sum_chunk = torch.sum(mag_chunk, dim=1)        # (chunk, num_targets)
            radar_3d_cpu[r_start:r_end] = sum_chunk.cpu().numpy()

        return radar_3d_cpu.reshape(num_range, self.num_az, self.num_el)  # (512, 751, 11)


# ── Extraction logic ──────────────────────────────────────────────────────────

def extract(args):
    # ── Validate required args ─────────────────────────────────────────────────
    missing = [(n, v) for n, v in [
        ("--radial-root", args.radial_root),
        ("--eval-index",  args.eval_index),
        ("--eval-split",  args.eval_split),
    ] if v is None]
    if missing:
        for name, _ in missing:
            print(f"ERROR: {name} is required.")
        print("See README.md — Data Preparation section.")
        sys.exit(1)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    radial_root = os.path.abspath(args.radial_root)
    _add_dbreader_to_path(radial_root)

    try:
        from DBReader.DBReader import SyncReader
    except ImportError as e:
        print(f"ERROR: Cannot import SyncReader — check --radial-root path. {e}")
        sys.exit(1)

    # ── Load index / split tables ─────────────────────────────────────────────
    df_idx   = pd.read_csv(args.eval_index)   # echo_frame_id, seq_name, local_frame_idx
    df_split = pd.read_csv(args.eval_split)   # echo_frame_id, split

    test_ids = set(df_split.loc[df_split["split"] == args.split, "echo_frame_id"].tolist())
    df_test  = df_idx[df_idx["echo_frame_id"].isin(test_ids)].copy()
    df_test  = df_test.sort_values("echo_frame_id").reset_index(drop=True)

    print(f"Found {len(df_test)} frames in split='{args.split}'")

    if args.dry_run:
        print(df_test.to_string())
        return

    # ── Output directories ────────────────────────────────────────────────────
    out_dir    = os.path.abspath(args.out_dir)
    cam_dir    = os.path.join(out_dir, "camera")
    radar_dir  = os.path.join(out_dir, "radar_FFT")
    os.makedirs(cam_dir,   exist_ok=True)
    os.makedirs(radar_dir, exist_ok=True)

    # ── Init RadarProcessor ───────────────────────────────────────────────────
    calib_path = os.path.join(radial_root, "SignalProcessing", "CalibrationTable.npy")
    device     = "cpu" if args.cpu else "cuda"
    processor  = RadarProcessor(calib_path, device=device)

    # ── Group by sequence to avoid re-opening SyncReader per frame ────────────
    seq_groups = df_test.groupby("seq_name")
    records    = []

    for seq_name, group in seq_groups:
        seq_path = os.path.join(radial_root, seq_name)
        if not os.path.isdir(seq_path):
            print(f"WARNING: Sequence not found — skipping: {seq_path}")
            continue

        print(f"\n[{seq_name}]  {len(group)} frames")
        try:
            # tolerance=20000 µs matches the sensor synchronisation window
            # used throughout the On-RADIal pipeline (camera @ ~10 Hz, radar @ ~25 Hz)
            db = SyncReader(seq_path, tolerance=20000, silent=True)
        except Exception as e:
            print(f"  Cannot open SyncReader: {e} — skipping")
            continue

        for _, row in tqdm(group.iterrows(), total=len(group), desc=seq_name[:20]):
            echo_id    = int(row["echo_frame_id"])
            local_idx  = int(row["local_frame_idx"])

            img_name   = f"{echo_id:06d}.jpg"
            npy_name   = f"{echo_id:06d}.npy"
            img_path   = os.path.join(cam_dir,   img_name)
            npy_path   = os.path.join(radar_dir, npy_name)

            if args.skip_existing and os.path.exists(npy_path) and os.path.exists(img_path):
                records.append({
                    "echo_frame_id":   echo_id,
                    "image":           os.path.join("camera",    img_name),
                    "radar":           os.path.join("radar_FFT", npy_name),
                    "seq_name":        seq_name,
                    "local_frame_idx": local_idx,
                })
                continue

            # ── Get raw sensor data ────────────────────────────────────────
            try:
                data = db.GetSensorData(local_idx)
            except Exception as e:
                print(f"  Frame {echo_id} (local {local_idx}): GetSensorData failed — {e}")
                continue

            # ── Camera ────────────────────────────────────────────────────
            cam_data = data.get("camera", {}).get("data")
            if cam_data is None:
                print(f"  Frame {echo_id}: no camera data")
                continue
            # Save in BGR (cv2 native) — consistent with create_dense_dataset.py
            cv2.imwrite(img_path, cam_data)

            # ── Radar ─────────────────────────────────────────────────────
            # Extract only the four ADC channels (same pattern used in On-RADIal)
            adc_data = {k: data[k] for k in ["radar_ch0", "radar_ch1", "radar_ch2", "radar_ch3"]}
            try:
                ra_map = processor.process_frame(adc_data)  # (512, 751, 11)
                np.save(npy_path, ra_map.astype(np.float32))
            except Exception as e:
                print(f"  Frame {echo_id}: RadarProcessor failed — {e}")
                continue

            records.append({
                "echo_frame_id":   echo_id,
                "image":           os.path.join("camera",    img_name),
                "radar":           os.path.join("radar_FFT", npy_name),
                "seq_name":        seq_name,
                "local_frame_idx": local_idx,
            })

    # ── Write index ───────────────────────────────────────────────────────────
    index_path = os.path.join(out_dir, "index.csv")
    pd.DataFrame(records).to_csv(index_path, index=False)
    print(f"\nExtracted {len(records)} / {len(df_test)} frames")
    print(f"Camera images -> {cam_dir}")
    print(f"Radar FFT maps -> {radar_dir}")
    print(f"Index CSV      -> {index_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract camera + Bartlett RA maps for EchoFusion test frames"
    )
    p.add_argument(
        "--radial-root",
        default=None,
        help="Path to the RADIal/ folder (contains SignalProcessing/ and RECORD@* dirs). REQUIRED.",
    )
    p.add_argument(
        "--eval-index",
        default=None,
        help="eval_index.csv with columns: echo_frame_id, seq_name, local_frame_idx. REQUIRED.",
    )
    p.add_argument(
        "--eval-split",
        default=None,
        help="eval_split.csv with columns: echo_frame_id, split. REQUIRED.",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join(_RA_SYNTHGEN_ROOT, "data", "radial"),
        help="Output directory (will contain camera/ radar_FFT/ index.csv)",
    )
    p.add_argument(
        "--split",
        default="test",
        help="Which split to extract (default: test)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print frame list without extracting anything",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip frames whose output files already exist (for resuming)",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU beamforming (no CUDA required, but ~10× slower)",
    )
    args = p.parse_args()
    extract(args)
