"""
RADIal selective extractor.

The RADIal.zip uses ZIP64 multi-disk format which Python's zipfile cannot open.
This script delegates to 7-Zip (preferred) or PowerShell as fallback.

Usage:
    python data/extract_radial.py --zip C:/path/to/RADIal.zip --out data/radial
    python data/extract_radial.py --zip C:/path/to/RADIal.zip --out data/radial --list-only
"""

import argparse
import os
import subprocess
import sys

# Folders/files to extract from RADIal.zip
WANTED_PREFIXES = (
    "img/",
    "fft_data/",
    "calib/",
    "labels.csv",
    "gt_labels.csv",
    "gt_calib.csv",
)

# Common 7z installation paths on Windows
SEVENZIP_CANDIDATES = [
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
    "7z",   # if it's on PATH
    "7za",  # standalone binary
]


def find_7z():
    for candidate in SEVENZIP_CANDIDATES:
        try:
            result = subprocess.run(
                [candidate, "i"], capture_output=True, timeout=5
            )
            if result.returncode == 0 or b"7-Zip" in result.stdout:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def list_contents_7z(z7, zip_path):
    """Return list of file names inside the archive."""
    result = subprocess.run(
        [z7, "l", "-ba", zip_path],
        capture_output=True,
        text=True,
    )
    names = []
    for line in result.stdout.splitlines():
        # 7z -ba output format: "date  time  attr  size  compressed  name"
        parts = line.strip().split()
        if len(parts) >= 6:
            names.append(parts[-1])
    return names


def extract_7z(z7, zip_path, out_dir, patterns):
    """
    Extract only entries matching the given prefix patterns using 7z.
    7z supports wildcard include/exclude with -i!pattern.
    """
    os.makedirs(out_dir, exist_ok=True)
    includes = []
    for p in patterns:
        # 7z wildcard: -i!prefix* for directories, -i!filename for files
        if p.endswith("/"):
            includes += [f"-i!{p}*"]
        else:
            includes += [f"-i!{p}"]

    cmd = [z7, "x", zip_path, f"-o{out_dir}", "-y"] + includes
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def extract_powershell(zip_path, out_dir, patterns):
    """
    Fallback: extract with PowerShell Expand-Archive (extracts everything,
    then prune unwanted folders). Only practical for smaller zips.
    """
    print("7-Zip not found. Attempting PowerShell Expand-Archive (extracts all)...")
    ps_cmd = (
        f"Expand-Archive -Path '{zip_path}' -DestinationPath '{out_dir}' -Force"
    )
    result = subprocess.run(
        ["powershell", "-Command", ps_cmd],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("PowerShell extraction failed:", result.stderr)
        return False
    print("Full extraction complete. Pruning unwanted folders...")
    # Remove folders that are not in WANTED_PREFIXES
    for entry in os.listdir(out_dir):
        full = os.path.join(out_dir, entry)
        keep = any(
            (entry + "/").startswith(p.rstrip("/")) or entry == p.rstrip("/")
            for p in patterns
        )
        if not keep and os.path.isdir(full):
            import shutil
            shutil.rmtree(full)
            print(f"  Removed: {full}")
    return True


def print_manual_instructions(zip_path, out_dir, patterns):
    print("\n" + "=" * 60)
    print("MANUAL EXTRACTION INSTRUCTIONS")
    print("=" * 60)
    print("Neither 7-Zip nor PowerShell could extract the archive.")
    print()
    print("Option 1 — Install 7-Zip (recommended, free):")
    print("  https://www.7-zip.org/download.html")
    print("  Then run this script again.")
    print()
    print("Option 2 — Use 7-Zip GUI:")
    print(f"  1. Open 7-Zip File Manager")
    print(f"  2. Navigate to: {zip_path}")
    print(f"  3. Extract ONLY these folders to {out_dir}:")
    for p in patterns:
        print(f"     - {p}")
    print()
    print("Option 3 — Windows Explorer:")
    print(f"  Right-click {zip_path} → Extract All → {out_dir}")
    print("  (This extracts everything; will use ~5-15 GB)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Selective RADIal.zip extractor")
    parser.add_argument(
        "--zip",
        default=r"C:\Users\Ilyes\Downloads\RADIal.zip",
        help="Path to RADIal.zip",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "radial"),
        help="Output directory",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Just list the archive contents, do not extract",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract everything (not just images/fft_data/calib)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.zip):
        print(f"Error: zip not found at {args.zip}")
        sys.exit(1)

    patterns = WANTED_PREFIXES if not args.all else ["*"]

    z7 = find_7z()

    if args.list_only:
        if z7:
            names = list_contents_7z(z7, args.zip)
            tops = sorted(set(n.split("/")[0] for n in names if n))
            print(f"Top-level entries ({len(names)} total):")
            for t in tops:
                print(f"  {t}/")
        else:
            print("7-Zip required to list contents. Install from https://7-zip.org")
        return

    print(f"Source  : {args.zip}")
    print(f"Output  : {args.out}")
    print(f"Patterns: {patterns}")
    print()

    if z7:
        print(f"Using 7-Zip: {z7}")
        ok = extract_7z(z7, args.zip, args.out, list(patterns))
        if ok:
            print(f"\nExtraction complete → {args.out}")
        else:
            print("7-Zip extraction reported an error.")
    else:
        ok = extract_powershell(args.zip, args.out, list(patterns))
        if not ok:
            print_manual_instructions(args.zip, args.out, list(patterns))
            sys.exit(1)
        else:
            print(f"\nExtraction complete → {args.out}")


if __name__ == "__main__":
    main()
