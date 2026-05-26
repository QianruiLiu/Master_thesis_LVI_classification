"""
Convert pre-extracted GigaPath tile embeddings from HDF5 format into the
per-slide .pt feature format expected by CLAM.

Usage
-----
Run from the project root directory:

python models/CLAM/1_h52pt.py \
  --h5-root /mnt/d/tile_encoder_h5files \
  --pt-root /mnt/d/CLAM_ptfiles

Outputs
-------
The script writes one CLAM-compatible .pt file per slide:
    <pt-root>/<slide_id>.pt
"""

import argparse
import os
from pathlib import Path

import h5py
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert GigaPath H5 tile embeddings to CLAM-compatible .pt files."
    )
    parser.add_argument(
        "--h5-root",
        type=Path,
        default=Path("/mnt/d/tile_encoder_h5files"),
        help="Input folder containing .h5 files with a 'tile_embeds' dataset. "
             "Default: /mnt/d/tile_encoder_h5files",
    )
    parser.add_argument(
        "--pt-root",
        type=Path,
        default=Path("/mnt/d/CLAM_ptfiles"),
        help="Output folder for CLAM-compatible .pt feature files. "
             "Default: /mnt/d/CLAM_ptfiles",
    )
    return parser.parse_args()


def convert_h5_to_pt(h5_root: Path, pt_root: Path, overwrite: bool = False):
    if not h5_root.exists():
        raise FileNotFoundError(f"h5_root does not exist: {h5_root}")
    if not h5_root.is_dir():
        raise NotADirectoryError(f"h5_root is not a directory: {h5_root}")

    pt_root.mkdir(parents=True, exist_ok=True)

    h5_files = sorted([p for p in h5_root.iterdir() if p.name.endswith(".h5")])
    print(f"Found H5 files: {len(h5_files)}")

    n_done = 0
    n_skipped = 0
    n_failed = 0

    for h5_path in h5_files:
        slide_id = h5_path.stem
        pt_path = pt_root / f"{slide_id}.pt"

        if pt_path.exists() and not overwrite:
            print(f"[SKIP] output exists: {pt_path}")
            n_skipped += 1
            continue

        try:
            with h5py.File(h5_path, "r") as f:
                if "tile_embeds" not in f:
                    raise KeyError(f"'tile_embeds' dataset not found in {h5_path}")
                feats = f["tile_embeds"][:]  # shape [N, embedding_dim]

            feats = torch.tensor(feats, dtype=torch.float32)
            torch.save(feats, pt_path)

            print(f"[DONE] {h5_path.name} -> {pt_path}  shape={tuple(feats.shape)}")
            n_done += 1

        except Exception as e:
            print(f"[FAIL] {h5_path.name}: {repr(e)}")
            n_failed += 1

    print("\nSummary:")
    print(f"  converted: {n_done}")
    print(f"  skipped:   {n_skipped}")
    print(f"  failed:    {n_failed}")
    print(f"  output:    {pt_root}")


def main():
    args = parse_args()
    convert_h5_to_pt(
        h5_root=args.h5_root,
        pt_root=args.pt_root
    )


if __name__ == "__main__":
    main()
