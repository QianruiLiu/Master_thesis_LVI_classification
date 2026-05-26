#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script extracts a foreground ROI crop from one whole-slide image and saves
both the ROI image and its coordinate metadata. It uses the same LoadROId-based
foreground extraction logic as the GigaPath preprocessing pipeline, so the saved
ROI crop can later be used as a background image for occlusion heatmap overlays.

The script records the ROI origin, scale, image size, preprocessing level, margin,
and foreground threshold. These metadata values are required to map global
level-0 slide coordinates back onto the ROI PNG during heatmap visualization.

Usage
-----
Example:
python 4_extract_cutting_figures.py \
    --slide_path /mnt/d/BMM_LVI/00PH05780.ndpi \
    --slide_id 00PH05780.ndpi \
    --out_dir /mnt/d/roi_pngs \
    --level 1 \
    --margin 0

Outputs
-------
For each input slide, the script writes:
    - <slide_id>_roi.png
        Foreground ROI crop image.

    - <slide_id>_roi_meta.json
        Metadata required for coordinate mapping, including origin_x, origin_y,
        scale, roi_width, and roi_height.

Notes
-----
This script does not train or evaluate a model. Its purpose is to prepare
visualization assets for later occlusion-based heatmap rendering. The level,
margin, and foreground threshold should be consistent with the preprocessing
settings used for the main pipeline.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from monai.data.wsi_reader import WSIReader
from gigapath.preprocessing.data.foreground_segmentation import LoadROId


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide_path", type=str, required=True, help="Path to .ndpi/.svs/.tiff slide")
    ap.add_argument("--slide_id", type=str, required=True, help="Slide ID, e.g. 02PL10819.ndpi")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--level", type=int, default=1, help="Same level used in preprocessing")
    ap.add_argument("--margin", type=int, default=0, help="Same margin used in preprocessing")
    ap.add_argument("--foreground_threshold", type=float, default=None,
                    help="Same foreground threshold used in preprocessing; omit for auto")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = {
        "slide_id": args.slide_id,
        "image": Path(args.slide_path),
        "metadata": {},
    }

    loader = LoadROId(
        WSIReader(backend="OpenSlide"),
        level=args.level,
        margin=args.margin,
        foreground_threshold=args.foreground_threshold,
    )

    sample = loader(sample)

    roi_img = sample["image"]          # (C, H, W)
    origin = sample["origin"]          # level-0 top-left of ROI crop
    scale = sample["scale"]            # mapping from ROI image coords to level-0
    fg_thr = sample.get("foreground_threshold", None)

    # save ROI png
    plt.figure(figsize=(8, 8))
    plt.imshow(roi_img.transpose(1, 2, 0))
    plt.axis("off")
    roi_png = out_dir / f"{args.slide_id}_roi.png"
    plt.savefig(roi_png, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    # save metadata
    meta = {
        "slide_id": str(args.slide_id),
        "slide_path": str(args.slide_path),
        "level": int(args.level),
        "margin": int(args.margin),
        "foreground_threshold": None if fg_thr is None else float(fg_thr),
        "origin_x": int(origin[0]),
        "origin_y": int(origin[1]),
        "scale": float(scale),
        "roi_height": int(roi_img.shape[1]),
        "roi_width": int(roi_img.shape[2]),
    }

    meta_json = out_dir / f"{args.slide_id}_roi_meta.json"
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] saved ROI image: {roi_png}")
    print(f"[DONE] saved ROI meta : {meta_json}")
    print(f"[INFO] origin = ({origin[0]}, {origin[1]})")
    print(f"[INFO] scale  = {scale}")


if __name__ == "__main__":
    main()