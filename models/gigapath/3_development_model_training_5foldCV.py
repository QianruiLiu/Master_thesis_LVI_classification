#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script trains and evaluates a slide-level LVI classifier based on
pre-extracted GigaPath tile embeddings. For each whole-slide image, tile-level
embeddings are read from an HDF5 file and passed through a frozen pretrained
GigaPath slide encoder. Only a linear classification head is trained to predict
slide-level LVI status.

The script supports either:
    1. a single train/validation/test split, or
    2. stratified K-fold cross-validation.

Usage
-----
Example: 5-fold cross-validation with ROI-priority training and heatmaps
The example usage can be seen in run_tuning_scripts/development_training_5_fold_CV.sh

Outputs:
- out_dir/fold_XX/{tb, best.pt, config.json, fold_metrics.json} + out_dir/cv_summary.json
- feature-importance heatmaps for selected TP, FP, TN, and FN examples (if --make heatmaps)
- 5_fold CV ROC curve
"""

import os
import math
import json
import random
import zlib
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

import openslide
import xml.etree.ElementTree as ET

import gigapath.slide_encoder as slide_encoder


# -----------------------------
# Utilities
# General helper functions used throughout the script.
# These include random seed control, label value cleaning,
# HDF5 feature loading, GigaPath slide encoder loading,
# and a wrapper for handling different model output formats.
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Convert label values safely into integers.
# Missing or empty values are treated as 0.
def safe_int01(x):
    if pd.isna(x):
        return 0
    try:
        return int(float(x))
    except Exception:
        s = str(x).strip()
        if s == "":
            return 0
        return int(s)


def read_h5_tile_data(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tile_embeds, coords, dist_to_roi).

    tile_embeds: [N, 1536]
    coords:      [N, 2]
    dist_to_roi: [N], NaN if not available
    """
    with h5py.File(h5_path, "r") as f:
        tile_embeds = f["tile_embeds"][:]   # [N,1536]
        coords = f["coords"][:]             # [N,2]
        dist_to_roi = f["dist_to_roi"][:] if "dist_to_roi" in f else np.full((coords.shape[0],), np.nan, np.float32)
    return tile_embeds, coords, dist_to_roi


def load_slide_encoder(global_pool: bool = True) -> nn.Module:
    """Load pretrained GigaPath slide encoder from HF hub."""
    # NOTE: CLS token not trained during pretraining; global_pool=True recommended.
    model = slide_encoder.create_model(
        "hf_hub:prov-gigapath/prov-gigapath",
        "gigapath_slide_enc12l768d",
        1536,
        global_pool=global_pool,
    )
    return model


def forward_slide_encoder(model: nn.Module, tile_embeds: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Robustly handle different return types from slide encoder."""
    out = model(tile_embeds, coords, all_layer_embed=False)
    if isinstance(out, (list, tuple)):
        out = out[-1]
    elif isinstance(out, dict):
        out = out.get("last_layer_embed", list(out.values())[-1])
    return out  # [B, E]


# -----------------------------
# ROI radius cache (read from NDPA + OpenSlide metadata)
# -----------------------------

def parse_circle_roi_ndpa(ndpa_path: Path) -> Optional[Tuple[float, float, float]]:
    """Parse the FIRST circle ROI in NDPA. Returns (cx_stage_nm, cy_stage_nm, r_nm) or None."""
    if not ndpa_path.exists():
        return None
    tree = ET.parse(str(ndpa_path))
    root = tree.getroot()
    circ = root.find(".//annotation[@type='circle']")
    if circ is None:
        return None
    cx_nm = float(circ.findtext("x"))
    cy_nm = float(circ.findtext("y"))
    r_nm = float(circ.findtext("radius"))
    return cx_nm, cy_nm, r_nm


def get_mpp(slide: openslide.OpenSlide) -> Tuple[float, float]:
    props = slide.properties
    mpp_x = props.get("openslide.mpp-x", None)
    mpp_y = props.get("openslide.mpp-y", None)
    if mpp_x is None or mpp_y is None:
        raise ValueError("Missing openslide.mpp-x / openslide.mpp-y.")
    return float(mpp_x), float(mpp_y)


def ndpa_stage_nm_to_level0_px(
    cx_stage_nm: float,
    cy_stage_nm: float,
    r_nm: float,
    slide: openslide.OpenSlide,
) -> Tuple[float, float, float]:
    """Convert Hamamatsu NDPA stage coordinates (nm) -> level-0 pixel coords.

    Uses Hamamatsu offsets:
      - hamamatsu.XOffsetFromSlideCentre
      - hamamatsu.YOffsetFromSlideCentre

    Assumes stage y-axis opposite to image y-axis (flip y).
    """
    props = slide.properties
    mpp_x, mpp_y = get_mpp(slide)

    x0_nm = float(props["hamamatsu.XOffsetFromSlideCentre"])
    y0_nm = float(props["hamamatsu.YOffsetFromSlideCentre"])

    W0, H0 = slide.dimensions
    W_nm = W0 * mpp_x * 1000.0
    H_nm = H0 * mpp_y * 1000.0

    x_rel = cx_stage_nm - x0_nm
    y_rel = cy_stage_nm - y0_nm

    x_img_nm = x_rel + (W_nm / 2.0)
    y_img_nm = y_rel + (H_nm / 2.0)

    x_px0 = x_img_nm / (mpp_x * 1000.0)
    y_px0 = y_img_nm / (mpp_y * 1000.0)

    r_px0 = r_nm / (mpp_x * 1000.0)  # approx with x mpp
    return float(x_px0), float(y_px0), float(r_px0)


@dataclass
class SlideItem:
    slide_id: str
    h5_path: Path
    ndpi_path: Path
    ndpa_path: Path
    y: int        # LVI(Ute)
    roi: int      # ROI flag (Johannes)


class RoiRadiusCache:
    """Cache ROI radius in level-0 pixels for slides that have a circle ROI."""

    def __init__(self):
        self.cache: Dict[str, float] = {}

    def get_roi_radius_px0(self, item: SlideItem) -> Optional[float]:
        if item.slide_id in self.cache:
            r = self.cache[item.slide_id]
            return None if (not np.isfinite(r)) else float(r)

        roi = parse_circle_roi_ndpa(item.ndpa_path)
        if roi is None:
            self.cache[item.slide_id] = float("nan")
            return None

        cx_nm, cy_nm, r_nm = roi
        slide = openslide.OpenSlide(str(item.ndpi_path))
        try:
            _, _, r_px0 = ndpa_stage_nm_to_level0_px(cx_nm, cy_nm, r_nm, slide)
        finally:
            slide.close()

        self.cache[item.slide_id] = float(r_px0)
        return float(r_px0)

def get_circle_roi_px0(item: SlideItem) -> Optional[Tuple[float, float, float]]:
    roi = parse_circle_roi_ndpa(item.ndpa_path)
    if roi is None:
        return None

    cx_nm, cy_nm, r_nm = roi
    slide = openslide.OpenSlide(str(item.ndpi_path))
    try:
        cx_px0, cy_px0, r_px0 = ndpa_stage_nm_to_level0_px(cx_nm, cy_nm, r_nm, slide)
    finally:
        slide.close()

    return float(cx_px0), float(cy_px0), float(r_px0)


# -----------------------------
# ROI PNG / meta helpers (mirrors script-1 logic)
# -----------------------------

def load_roi_meta_for_slide(roi_png_dir: Path, slide_id: str) -> Optional[dict]:
    """
    Look for {roi_png_dir}/{slide_id}_roi_meta (JSON, no extension) and parse it.
    Returns None if the file does not exist or is missing required keys.
    Required keys: origin_x, origin_y, scale, roi_width, roi_height.
    """
    meta_path = roi_png_dir / f"{slide_id}_roi_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    required = ["origin_x", "origin_y", "scale", "roi_width", "roi_height"]
    if any(k not in meta for k in required):
        return None
    return meta


def load_roi_png_for_slide(
    roi_png_dir: Path,
    slide_id: str,
    thumb_max_px: int,
):
    """
    Load {roi_png_dir}/{slide_id}_roi (PNG, no extension) and downscale to thumb_max_px.
    Returns (pil_img, roi_w, roi_h, thumb_w, thumb_h) or None on failure.
    roi_w / roi_h are the *display* pixel dimensions of the PNG before downscaling.
    They must be overridden by roi_meta['roi_width'] / roi_meta['roi_height'] afterwards
    (those are the level-0-based ROI dimensions used for coordinate mapping).
    """
    png_path  = roi_png_dir / f"{slide_id}_roi.png"
    if not png_path.exists():
        return None
    try:
        from PIL import Image as _Image
        _Image.MAX_IMAGE_PIXELS = None
        img = _Image.open(png_path).convert("RGB")
        disp_w, disp_h = img.size
        scale_disp = float(thumb_max_px) / float(max(disp_w, disp_h))
        if scale_disp < 1.0:
            tw = max(1, int(round(disp_w * scale_disp)))
            th = max(1, int(round(disp_h * scale_disp)))
            img = img.resize((tw, th), _Image.LANCZOS)
        thumb_w, thumb_h = img.size
        return img, disp_w, disp_h, thumb_w, thumb_h
    except Exception:
        return None


def has_roi_png_assets(roi_png_dir: Optional[Path], slide_id: str, thumb_max_px: int) -> Tuple[bool, str]:
    """Return (ok, reason) indicating whether both ROI PNG and ROI meta are usable."""
    if roi_png_dir is None:
        return False, "roi_png_dir is None"
    meta = load_roi_meta_for_slide(roi_png_dir, slide_id)
    if meta is None:
        return False, "missing or invalid ROI meta JSON"
    roi_png_result = load_roi_png_for_slide(roi_png_dir, slide_id, thumb_max_px)
    if roi_png_result is None:
        return False, "missing or unreadable ROI PNG"
    return True, "ok"


def classify_confusion_category(y_true: int, y_pred: int) -> str:
    y_true = int(y_true)
    y_pred = int(y_pred)
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def pick_extreme_confusion_examples(
    records: List[Dict],
    category: str,
    n_pick: int = 2,
) -> List[Dict]:
    """Pick up to n_pick slides from one confusion class.

    Preference rule requested by user:
      - TP / FP: prefer probability > 0.9, then highest probability.
      - TN / FN: prefer probability < 0.1, then lowest probability.
    If fewer than n_pick satisfy the strict preference, fill from the remaining
    slides of the same confusion class using the same extremeness ordering.
    """
    cat_records = [r for r in records if r["confusion"] == category and r.get("roi_png_ok", False)]
    if category in ("TP", "FP"):
        strict = [r for r in cat_records if np.isfinite(r["prob"]) and r["prob"] > 0.9]
        strict.sort(key=lambda r: (-r["prob"], r["slide_id"]))
        if len(strict) >= n_pick:
            return strict[:n_pick]
        remain_ids = {r["slide_id"] for r in strict}
        fallback = [r for r in cat_records if r["slide_id"] not in remain_ids]
        fallback.sort(key=lambda r: (-r["prob"], r["slide_id"]))
        return (strict + fallback)[:n_pick]
    else:
        strict = [r for r in cat_records if np.isfinite(r["prob"]) and r["prob"] < 0.1]
        strict.sort(key=lambda r: (r["prob"], r["slide_id"]))
        if len(strict) >= n_pick:
            return strict[:n_pick]
        remain_ids = {r["slide_id"] for r in strict}
        fallback = [r for r in cat_records if r["slide_id"] not in remain_ids]
        fallback.sort(key=lambda r: (r["prob"], r["slide_id"]))
        return (strict + fallback)[:n_pick]


def generate_confusion_heatmaps_for_split(
    split_items: List[SlideItem],
    model: nn.Module,
    head: nn.Module,
    device: torch.device,
    threshold: float,
    args,
    out_dir: Path,
) -> Dict[str, object]:
    """Generate heatmaps for TP/TN/FP/FN examples on one split.

    Uses the same selected threshold as sensitivity/specificity calculation.
    Requires ROI PNG assets; slides missing ROI PNG/meta are skipped and logged.
    """
    hm_dir = out_dir / "heatmaps"
    hm_dir.mkdir(parents=True, exist_ok=True)

    split_name = args.heatmap_split
    n_pred_samples = args.eval_samples_test if split_name == "test" else args.eval_samples_val
    roi_png_dir = Path(args.roi_png_dir) if args.roi_png_dir else None

    records = []
    missing_roi_png = []
    available_roi_png = []

    for it in split_items:
        mean_prob = compute_slide_mean_probability(
            item=it,
            model=model,
            head=head,
            device=device,
            k_max=args.k_max,
            tile_size=args.tile_size,
            margin_px=args.margin_px,
            roi_frac=args.roi_frac,
            seed=args.seed + 777,
            n_samples=n_pred_samples,
        )
        if not np.isfinite(mean_prob):
            continue

        y_pred = int(mean_prob >= float(threshold))
        confusion = classify_confusion_category(it.y, y_pred)
        roi_ok, roi_reason = has_roi_png_assets(roi_png_dir, it.slide_id, args.heatmap_thumb_max_px)

        rec = {
            "slide_id": it.slide_id,
            "y_true": int(it.y),
            "y_pred": int(y_pred),
            "prob": float(mean_prob),
            "confusion": confusion,
            "roi_png_ok": bool(roi_ok),
            "roi_png_reason": str(roi_reason),
        }
        records.append(rec)

        if roi_ok:
            available_roi_png.append(it.slide_id)
            print(f"[HEATMAP][ROI_PNG][FOUND] {split_name} slide={it.slide_id} | class={confusion} | prob={mean_prob:.3f}")
        else:
            missing_roi_png.append(rec)
            print(f"[HEATMAP][ROI_PNG][SKIP] {split_name} slide={it.slide_id} | class={confusion} | prob={mean_prob:.3f} | reason={roi_reason}")

    selected = {}
    selected_ordered = []
    for cat in ["TP", "FP", "TN", "FN"]:
        picks = pick_extreme_confusion_examples(records, cat, n_pick=2)
        selected[cat] = picks
        if len(picks) == 0:
            print(f"[HEATMAP][SELECT] {cat}: no eligible slide with ROI PNG assets.")
        else:
            for r in picks:
                selected_ordered.append(r)
                print(f"[HEATMAP][SELECT] {cat}: slide={r['slide_id']} | prob={r['prob']:.3f} | y_true={r['y_true']} | y_pred={r['y_pred']}")

    slide_lookup = {it.slide_id: it for it in split_items}
    generated = []
    for r in selected_ordered:
        it = slide_lookup[r["slide_id"]]
        cat_dir = hm_dir / r["confusion"]
        cat_dir.mkdir(parents=True, exist_ok=True)
        print(f"[HEATMAP][RUN] {r['confusion']} {split_name} slide={it.slide_id} | prob={r['prob']:.3f}")
        res = make_occlusion_heatmap_one_slide_repeated(
            item=it,
            model=model,
            head=head,
            device=device,
            k_max=args.k_max,
            tile_size=args.tile_size,
            block_px=args.heatmap_block_px,
            refill=bool(args.heatmap_refill),
            seed=args.seed,
            out_dir=cat_dir,
            n_repeats=args.heatmap_repeats,
            thumb_max_px=args.heatmap_thumb_max_px,
            roi_png_dir=roi_png_dir,
            pred_label=int(r["y_pred"]),
            threshold=float(threshold),
        )
        if res is not None:
            res["confusion"] = r["confusion"]
            res["prob"] = float(r["prob"])
            res["y_true"] = int(r["y_true"])
            res["y_pred"] = int(r["y_pred"])
            generated.append(res)

    summary = {
        "threshold": float(threshold),
        "split": split_name,
        "available_roi_png_slides": available_roi_png,
        "missing_roi_png_slides": missing_roi_png,
        "selected_examples": selected,
        "generated_heatmaps": generated,
    }
    with open(hm_dir / "heatmap_selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary

def rect_circle_overlap(
    x0: float, y0: float, x1: float, y1: float,
    cx: float, cy: float, r: float
) -> bool:
    nearest_x = min(max(cx, x0), x1)
    nearest_y = min(max(cy, y0), y1)
    dx = cx - nearest_x
    dy = cy - nearest_y
    return (dx * dx + dy * dy) <= (r * r)

# -----------------------------
# Sampling
# This section controls which tiles are selected from each slide.
# Because slides can contain many tiles, each training/evaluation step uses
# at most k_max tiles per slide.
# -----------------------------

def sample_tile_indices(
    dist_to_roi: np.ndarray,
    n_tiles: int,
    k_max: int,
    tile_size: int,
    margin_px: float,
    roi_radius_px0: Optional[float],
    use_roi_sampling: bool,
    roi_frac: float,
    rng: np.random.RandomState,
    roi_sampling_mode: str = "near",
    roi_inside_cap: Optional[int] = None,
) -> np.ndarray:
    """Choose indices for a slide.

    K_total = min(k_max, n_tiles)

    Modes:
      - "near" (legacy):
          define near as dist_to_roi <= roi_radius + margin + tile_half_diag
          sample K_roi ~ roi_frac*K_total from near, remainder from global.
      - "inside_priority":
          define inside as dist_to_roi <= roi_radius (tile centre inside ROI).
          sample up to roi_inside_cap from inside ROI, remainder from global.
          If inside ROI has fewer tiles than the cap, take all inside tiles.
          Across epochs, random draws from large ROIs naturally rotate coverage.

    Else / fallback:
      - uniform random from all tiles
    """
    K_total = min(k_max, n_tiles)
    all_idx = np.arange(n_tiles, dtype=np.int64)

    # Uniform random fallback
    if (not use_roi_sampling) or (roi_radius_px0 is None) or (not np.isfinite(roi_radius_px0)):
        if n_tiles <= K_total:
            return all_idx
        return rng.choice(all_idx, size=K_total, replace=False)

    dist = dist_to_roi.astype(np.float32)
    if not np.all(np.isfinite(dist)):
        finite_mask = np.isfinite(dist)
        if finite_mask.sum() == 0:
            if n_tiles <= K_total:
                return all_idx
            return rng.choice(all_idx, size=K_total, replace=False)
        dist = dist.copy()
        dist[~finite_mask] = 1e12
    # ROI-priority mode:
    # first sample tiles whose centres fall inside the annotated ROI,
    # then fill the remaining tile budget with globally sampled tiles.
    if roi_sampling_mode == "inside_priority":
        inside_idx = all_idx[dist <= float(roi_radius_px0)]

        if inside_idx.size == 0:
            if n_tiles <= K_total:
                return all_idx
            return rng.choice(all_idx, size=K_total, replace=False)

        K_inside_cap = K_total if roi_inside_cap is None else int(max(0, roi_inside_cap))
        K_inside = min(K_total, K_inside_cap, inside_idx.size)

        chosen_roi = rng.choice(inside_idx, size=K_inside, replace=False) if inside_idx.size > K_inside else inside_idx

        remain = K_total - chosen_roi.size
        if remain <= 0:
            return chosen_roi

        mask = np.ones(n_tiles, dtype=bool)
        mask[chosen_roi] = False
        pool = all_idx[mask]
        if pool.size == 0:
            return chosen_roi
        if pool.size <= remain:
            chosen_bg = pool
        else:
            chosen_bg = rng.choice(pool, size=remain, replace=False)

        return np.concatenate([chosen_roi, chosen_bg], axis=0)

    tile_half_diag = (math.sqrt(2.0) * tile_size) / 2.0
    thr = float(roi_radius_px0) + float(margin_px) + float(tile_half_diag)
    near_idx = all_idx[dist <= thr]

    if near_idx.size == 0:
        if n_tiles <= K_total:
            return all_idx
        return rng.choice(all_idx, size=K_total, replace=False)

    K_roi_target = int(round(roi_frac * K_total))
    K_roi = min(K_roi_target, near_idx.size)

    chosen_roi = rng.choice(near_idx, size=K_roi, replace=False) if near_idx.size > K_roi else near_idx

    remain = K_total - chosen_roi.size
    if remain <= 0:
        return chosen_roi

    mask = np.ones(n_tiles, dtype=bool)
    mask[chosen_roi] = False
    pool = all_idx[mask]
    if pool.size == 0:
        return chosen_roi
    if pool.size <= remain:
        chosen_bg = pool
    else:
        chosen_bg = rng.choice(pool, size=remain, replace=False)

    return np.concatenate([chosen_roi, chosen_bg], axis=0)


# -----------------------------
# Metrics / Evaluation
# This section calculates model performance.
# AUC is used for threshold-independent evaluation.
# Sensitivity and specificity are calculated after selecting a probability threshold.
# ROC curves from multiple CV folds can also be averaged on a common FPR grid.
# -----------------------------

def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))

def compute_roc_points(y_true: np.ndarray, y_prob: np.ndarray):
    """Return ROC points for one prediction set.

    x-axis = false positive rate = 1 - specificity
    y-axis = true positive rate = sensitivity
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    finite_mask = np.isfinite(y_prob)
    y_true = y_true[finite_mask]
    y_prob = y_prob[finite_mask]

    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def average_roc_curves_across_folds(fold_results: List[Dict], num_points: int = 101) -> Optional[Dict]:
    """Average ROC curves from CV folds on a shared FPR grid."""
    common_fpr = np.linspace(0.0, 1.0, int(num_points), dtype=np.float32)
    tpr_list = []
    per_fold_curves = []

    for m in fold_results:
        y_true = np.asarray(m.get("test_y_true", []), dtype=np.int64)
        y_prob = np.asarray(m.get("test_y_prob", []), dtype=np.float32)

        roc_res = compute_roc_points(y_true, y_prob)
        if roc_res is None:
            continue

        fpr = np.asarray(roc_res["fpr"], dtype=np.float32)
        tpr = np.asarray(roc_res["tpr"], dtype=np.float32)

        interp_tpr = np.interp(common_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0

        tpr_list.append(interp_tpr)
        per_fold_curves.append({
            "fold": int(m.get("fold", len(per_fold_curves) + 1)),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        })

    if len(tpr_list) == 0:
        return None

    tpr_mat = np.asarray(tpr_list, dtype=np.float32)
    return {
        "n_folds_used": int(tpr_mat.shape[0]),
        "common_fpr": common_fpr.tolist(),
        "mean_tpr": np.mean(tpr_mat, axis=0).tolist(),
        "std_tpr": np.std(tpr_mat, axis=0).tolist(),
        "per_fold_curves": per_fold_curves,
    }


def plot_mean_roc_curve(
    roc_summary: Dict,
    out_png: Path,
    mean_auc: float = None,
    std_auc: float = None,
):
    import matplotlib.pyplot as plt

    fpr = np.asarray(roc_summary["common_fpr"], dtype=np.float32)
    mean_tpr = np.asarray(roc_summary["mean_tpr"], dtype=np.float32)
    std_tpr = np.asarray(roc_summary["std_tpr"], dtype=np.float32)

    lower = np.clip(mean_tpr - std_tpr, 0.0, 1.0)
    upper = np.clip(mean_tpr + std_tpr, 0.0, 1.0)

    if mean_auc is not None and std_auc is not None and np.isfinite(mean_auc) and np.isfinite(std_auc):
        roc_label = f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})"
    elif mean_auc is not None and np.isfinite(mean_auc):
        roc_label = f"Mean ROC (AUC = {mean_auc:.3f})"
    else:
        roc_label = "Mean ROC"

    plt.figure(figsize=(6, 6), dpi=150)
    plt.plot(fpr, mean_tpr, linewidth=2.0, label=roc_label)
    plt.fill_between(fpr, lower, upper, alpha=0.2, label="±1 std")
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, alpha=0.7)

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Mean ROC across CV folds")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def binary_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    return tp, fn, tn, fp


def sensitivity_specificity_from_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= float(threshold)).astype(np.int64)

    tp, fn, tn, fp = binary_confusion_counts(y_true, y_pred)

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    return {
        "threshold": float(threshold),
        "sensitivity": sens,
        "specificity": spec,
        "tp": int(tp),
        "fn": int(fn),
        "tn": int(tn),
        "fp": int(fp),
    }

# Select a classification threshold using the validation set.
# The main rule is:
#   1. keep thresholds that reach the target sensitivity, e.g. >= 0.75;
#   2. among them, choose the threshold with the highest specificity.

def select_threshold_for_target_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float = 0.75,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    finite_mask = np.isfinite(y_prob)
    y_true = y_true[finite_mask]
    y_prob = y_prob[finite_mask]

    if y_true.size == 0:
        return {
            "selected_threshold": float("nan"),
            "selected_val_sensitivity": float("nan"),
            "selected_val_specificity": float("nan"),
        }

    candidate_thresholds = sorted(set([0.0, 1.0] + [float(x) for x in y_prob.tolist()]))

    stats = []
    for thr in candidate_thresholds:
        m = sensitivity_specificity_from_threshold(y_true, y_prob, thr)
        stats.append(m)

    eligible = [
        m for m in stats
        if np.isfinite(m["sensitivity"]) and (m["sensitivity"] >= float(target_sensitivity))
    ]

    if len(eligible) > 0:
        # maximize specificity; if tied, choose higher threshold
        best = sorted(
            eligible,
            key=lambda m: (
                -m["specificity"] if np.isfinite(m["specificity"]) else float("inf"),
                -m["threshold"],
            ),
        )[0]
    else:
        # fallback: maximize sensitivity; if tied, specificity; if tied, threshold
        best = sorted(
            stats,
            key=lambda m: (
                -m["sensitivity"] if np.isfinite(m["sensitivity"]) else float("inf"),
                -m["specificity"] if np.isfinite(m["specificity"]) else float("inf"),
                -m["threshold"],
            ),
        )[0]

    return {
        "selected_threshold": float(best["threshold"]),
        "selected_val_sensitivity": float(best["sensitivity"]),
        "selected_val_specificity": float(best["specificity"]),
    }


@torch.no_grad()
def evaluate(
    items: List[SlideItem],
    model: nn.Module,
    head: nn.Module,
    device: torch.device,
    k_max: int,
    tile_size: int,
    margin_px: float,
    roi_frac: float,
    seed: int,
    n_samples: int,
) -> Dict[str, float]:
    """Uniform (non-ROI) evaluation with repeated sampling per slide."""
    model.eval()
    head.eval()

    y_true, y_prob = [], []
    losses = []

    bce = nn.BCEWithLogitsLoss()

    base_rng = np.random.RandomState(seed)

    for item in items:
        tile_embeds_np, coords_np, dist_np = read_h5_tile_data(item.h5_path)
        n_tiles = tile_embeds_np.shape[0]
        if n_tiles == 0:
            continue

        logits_list = []
        probs_list = []

        for _ in range(int(n_samples)):
            sub_seed = int(base_rng.randint(0, 2**31 - 1))
            rng = np.random.RandomState(sub_seed)

            idx = sample_tile_indices(
                dist_to_roi=dist_np,
                n_tiles=n_tiles,
                k_max=k_max,
                tile_size=tile_size,
                margin_px=margin_px,
                roi_radius_px0=None,
                use_roi_sampling=False,
                roi_frac=roi_frac,
                rng=rng,
            )

            emb = torch.from_numpy(tile_embeds_np[idx]).to(
                device=device,
                dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            coords = torch.from_numpy(coords_np[idx]).to(device=device, dtype=torch.float32)

            emb = emb.unsqueeze(0)
            coords = coords.unsqueeze(0)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(device.type == "cuda"),
                dtype=torch.float16,
            ):
                slide_vec = forward_slide_encoder(model, emb, coords)
                logit = head(slide_vec).squeeze(1)

            logit_scalar = float(logit.detach().cpu().numpy().reshape(-1)[0])
            logits_list.append(logit_scalar)
            probs_list.append(float(1.0 / (1.0 + np.exp(-logit_scalar))))

        mean_logit = float(np.mean(logits_list))
        mean_prob = float(np.mean(probs_list))

        y = torch.tensor([item.y], device=device, dtype=torch.float32)
        mean_logit_t = torch.tensor([mean_logit], device=device, dtype=torch.float32)
        loss = bce(mean_logit_t, y)
        losses.append(float(loss.item()))

        y_true.append(item.y)
        y_prob.append(mean_prob)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    return {
        "val_loss": float(np.mean(losses)) if len(losses) else float("nan"),
        "val_auc": compute_auc(y_true, y_prob),
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
    }


# -----------------------------
# Data building
# Build the slide dataset by matching three sources:
#   1. HDF5 tile embedding files from h5_root,
#   2. original NDPI slides from slides_dir,
#   3. slide-level ROI/LVI labels from labels_tsv.
# -----------------------------

def build_items(args) -> List[SlideItem]:
    """Build SlideItem list by matching H5 <-> NDPI <-> labels."""
    lab = pd.read_csv(args.labels_tsv, sep="\t")
    lab.columns = [c.strip() for c in lab.columns]
    assert set(["id", "ROI", "LVI"]).issubset(set(lab.columns)), (
        f"labels TSV must have columns: id ROI LVI. Got: {lab.columns.tolist()}"
    )

    lab["ROI"] = lab["ROI"].apply(safe_int01)
    lab["LVI"] = lab["LVI"].apply(safe_int01)

    h5_root = Path(args.h5_root)
    slides_dir = Path(args.slides_dir)

    h5_files = sorted(h5_root.glob("*.tile_embeds.h5"))
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No H5 files found in: {h5_root}")

    ndpi_map = {p.name: p for p in sorted(slides_dir.glob("*.ndpi"))}

    items: List[SlideItem] = []
    missing = 0
    for h5p in h5_files:
        slide_id = h5p.name.replace(".tile_embeds.h5", "")
        if slide_id not in ndpi_map:
            missing += 1
            continue

        row = lab[lab["id"] == slide_id]
        if row.shape[0] == 0:
            continue

        y = int(row["LVI"].values[0])
        roi = int(row["ROI"].values[0])

        ndpi_path = ndpi_map[slide_id]
        ndpa_path = Path(str(ndpi_path) + ".ndpa")

        items.append(
            SlideItem(
                slide_id=slide_id,
                h5_path=h5p,
                ndpi_path=ndpi_path,
                ndpa_path=ndpa_path,
                y=y,
                roi=roi,
            )
        )

    print(f"Found H5 files: {len(h5_files)}")
    print(f"Matched slides with NDPI + labels: {len(items)}  (missing NDPI for {missing} H5s)")
    return items


# -----------------------------
# Training (one fold)
# This section trains the model for one split/fold.
# The pretrained GigaPath slide encoder is frozen.
# Only the final linear classification head is updated.
# The best checkpoint is selected based on validation AUC.
# -----------------------------

def infer_embed_dim(model: nn.Module, device: torch.device) -> int:
    with torch.no_grad():
        dummy_emb = torch.randn(
            1, 4, 1536,
            device=device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        dummy_xy = torch.zeros(1, 4, 2, device=device, dtype=torch.float32)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
            dummy_out = forward_slide_encoder(model, dummy_emb, dummy_xy)
        return int(dummy_out.shape[-1])


# -----------------------------
# Occlusion / ablation heatmaps (feature importance)
# This section generates qualitative feature-importance maps.
# The idea is to remove one spatial block of sampled tiles at a time and
# measure how much the slide-level probability changes.
#
# For predicted positive slides:
#   important blocks are those whose removal decreases the predicted probability.
#
# For predicted negative slides:
#   important blocks are those whose removal increases the predicted probability.
#
# These maps should be interpreted as model-based importance maps, not as
# definitive lesion-level annotations.
# -----------------------------

def _stable_int_from_str(s: str) -> int:
    """Stable non-cryptographic hash -> int (for deterministic RNG seeds)."""
    return int(zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF)


@torch.no_grad()
def predict_logit_for_indices(
    model: nn.Module,
    head: nn.Module,
    tile_embeds_np: np.ndarray,
    coords_np: np.ndarray,
    idx: np.ndarray,
    device: torch.device,
) -> float:
    """Return slide-level logit for a chosen subset of tiles."""
    if idx.size == 0:
        return float("nan")

    emb = torch.from_numpy(tile_embeds_np[idx]).to(
        device=device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    coords = torch.from_numpy(coords_np[idx]).to(
        device=device,
        dtype=torch.float32,
    )

    emb = emb.unsqueeze(0)
    coords = coords.unsqueeze(0)

    with torch.amp.autocast(
        device_type="cuda",
        enabled=(device.type == "cuda"),
        dtype=torch.float16,
    ):
        slide_vec = forward_slide_encoder(model, emb, coords)
        logit = head(slide_vec).squeeze(1)

    return float(logit.detach().cpu().numpy().reshape(-1)[0])


@torch.no_grad()
def compute_slide_mean_probability(
    item: SlideItem,
    model: nn.Module,
    head: nn.Module,
    device: torch.device,
    k_max: int,
    tile_size: int,
    margin_px: float,
    roi_frac: float,
    seed: int,
    n_samples: int,
) -> float:
    """Repeated uniform sampling prediction for one slide; returns mean LVI probability."""
    tile_embeds_np, coords_np, dist_np = read_h5_tile_data(item.h5_path)
    n_tiles = tile_embeds_np.shape[0]
    if n_tiles == 0:
        return float("nan")

    base_seed = int(seed) + _stable_int_from_str(item.slide_id) % 1000003
    probs = []

    for r in range(int(n_samples)):
        rng = np.random.RandomState(base_seed + int(r) * 10007)

        idx = sample_tile_indices(
            dist_to_roi=dist_np,
            n_tiles=n_tiles,
            k_max=k_max,
            tile_size=tile_size,
            margin_px=margin_px,
            roi_radius_px0=None,
            use_roi_sampling=False,
            roi_frac=roi_frac,
            rng=rng,
        )
        logit = predict_logit_for_indices(model, head, tile_embeds_np, coords_np, idx, device)
        if np.isfinite(logit):
            probs.append(float(1.0 / (1.0 + np.exp(-logit))))

    if len(probs) == 0:
        return float("nan")
    return float(np.mean(np.asarray(probs, dtype=np.float32)))



def compute_occlusion_blocks_one_repeat(
    item: "SlideItem",
    model: nn.Module,
    head: nn.Module,
    device: torch.device,
    k_max: int,
    tile_size: int,
    block_px: int,
    refill: bool,
    seed: int,
    repeat_idx: int,
) -> Optional[Dict]:
    """
    One repeat of sampled-subset occlusion heatmap.
    Returns a dict with:
      - prob_base
      - block_scores_drop: {(bxi, byi): probability_drop}
      - block_scores_increase: {(bxi, byi): probability_increase}
      - block_drop_prob_raw: {(bxi, byi): probability_drop}
      - block_increase_prob_raw: {(bxi, byi): probability_increase}
      - block_counts: {(bxi, byi): n_tiles_in_block}
    """
    tile_embeds_np, coords_np, dist_np = read_h5_tile_data(item.h5_path)
    n_tiles = tile_embeds_np.shape[0]
    if n_tiles == 0:
        return None

    K_total = min(int(k_max), int(n_tiles))

    # use a repeat-specific deterministic seed
    base_seed = int(seed) + _stable_int_from_str(item.slide_id) % 1000003 + int(repeat_idx) * 10007
    rng = np.random.RandomState(base_seed)

    idx_base = sample_tile_indices(
        dist_to_roi=dist_np,
        n_tiles=n_tiles,
        k_max=k_max,
        tile_size=tile_size,
        margin_px=0.0,
        roi_radius_px0=None,
        use_roi_sampling=False,
        roi_frac=0.0,
        rng=rng,
    )
    idx_base = np.array(idx_base, dtype=np.int64)
    if idx_base.size == 0:
        return None

    logit_base = predict_logit_for_indices(model, head, tile_embeds_np, coords_np, idx_base, device)
    if not np.isfinite(logit_base):
        return None
    prob_base = float(1.0 / (1.0 + np.exp(-logit_base)))

    coords_sel = coords_np[idx_base]

    tile_footprint_px0 = 2048.0

    cx_sel = coords_sel[:, 0] + tile_footprint_px0 / 2.0
    cy_sel = coords_sel[:, 1] + tile_footprint_px0 / 2.0

    bx = np.floor_divide(cx_sel.astype(np.int64), int(block_px))
    by = np.floor_divide(cy_sel.astype(np.int64), int(block_px))

    block_pairs = np.stack([bx, by], axis=1)
    uniq_pairs = np.unique(block_pairs, axis=0)

    all_idx = np.arange(n_tiles, dtype=np.int64)

    block_scores_drop = {}
    block_scores_increase = {}
    block_drop_prob_raw = {}
    block_increase_prob_raw = {}
    block_counts = {}
    # group the selected tiles into large spatial blocks.
    # For each block, remove that block and re-run the model.
    for bxi, byi in uniq_pairs:
        mask_block = (bx == bxi) & (by == byi)
        idx_block = idx_base[mask_block]
        idx_keep = idx_base[~mask_block]
        idx_occ = idx_keep

        if refill and idx_occ.size < K_total:
            need = int(K_total - idx_occ.size)
            removed_set = set(idx_block.tolist())
            keep_set = set(idx_occ.tolist())

            mask_pool = np.ones(n_tiles, dtype=bool)
            if len(removed_set) > 0:
                mask_pool[list(removed_set)] = False
            if len(keep_set) > 0:
                mask_pool[list(keep_set)] = False
            pool = all_idx[mask_pool]

            if pool.size > 0:
                rng_block = np.random.RandomState(base_seed + int(bxi) * 1009 + int(byi) * 9176)
                take = pool if pool.size <= need else rng_block.choice(pool, size=need, replace=False)
                idx_occ = np.concatenate([idx_occ, take.astype(np.int64)], axis=0)

        if idx_occ.size == 0:
            prob_occ = float("nan")
        else:
            logit_occ = predict_logit_for_indices(model, head, tile_embeds_np, coords_np, idx_occ, device)
            prob_occ = float(1.0 / (1.0 + np.exp(-logit_occ))) if np.isfinite(logit_occ) else float("nan")

        n_block = int(idx_block.size)
        drop_prob = float(prob_base - prob_occ) if (np.isfinite(prob_base) and np.isfinite(prob_occ)) else float("nan")
        increase_prob = float(prob_occ - prob_base) if (np.isfinite(prob_base) and np.isfinite(prob_occ)) else float("nan")

        key = (int(bxi), int(byi))
        block_scores_drop[key] = drop_prob
        block_scores_increase[key] = increase_prob
        block_drop_prob_raw[key] = drop_prob
        block_increase_prob_raw[key] = increase_prob
        block_counts[key] = n_block

    return {
        "prob_base": float(prob_base),
        "block_scores_drop": block_scores_drop,
        "block_scores_increase": block_scores_increase,
        "block_drop_prob_raw": block_drop_prob_raw,
        "block_increase_prob_raw": block_increase_prob_raw,
        "block_counts": block_counts,
    }


def evaluate_occlusion_localization_one_slide(
    item: SlideItem,
    npz_path: Path,
) -> Optional[Dict[str, float]]:
    roi_circle = get_circle_roi_px0(item)
    if roi_circle is None or (not npz_path.exists()):
        return None

    cx, cy, r = roi_circle
    data = np.load(npz_path, allow_pickle=True)

    if "mean_score_grid" in data:
        score_key = "mean_score_grid"
    elif "mean_drop_prob_grid" in data:
        score_key = "mean_drop_prob_grid"
    elif "mean_drop_prob_per_tile_grid" in data:
        score_key = "mean_drop_prob_per_tile_grid"
    else:
        score_key = "mean_drop_grid"
    score_grid = data[score_key]
    block_px = int(data["block_px"])
    bx_min = int(data["bx_min"])
    by_min = int(data["by_min"])

    ranked = []
    for gy in range(score_grid.shape[0]):
        for gx in range(score_grid.shape[1]):
            score = float(score_grid[gy, gx])
            if not np.isfinite(score):
                continue

            bxi = bx_min + gx
            byi = by_min + gy
            x0 = bxi * block_px
            y0 = byi * block_px
            x1 = x0 + block_px
            y1 = y0 + block_px

            overlap = rect_circle_overlap(x0, y0, x1, y1, cx, cy, r)
            ranked.append((score, overlap))

    if len(ranked) == 0:
        return None

    ranked.sort(key=lambda t: t[0], reverse=True)

    hit1 = 1 if ranked[0][1] else 0
    hit5 = 1 if any(t[1] for t in ranked[:5]) else 0

    return {
        "slide_id": item.slide_id,
        "hit_at_1": int(hit1),
        "hit_at_5": int(hit5),
    }


def evaluate_occlusion_localization_split(
    items: List[SlideItem],
    heatmap_dir: Path,
) -> Dict[str, float]:
    target_items = [it for it in items if int(it.y) == 1 and int(it.roi) == 1]
    per_slide = []

    for item in target_items:
        npz_path = heatmap_dir / f"{item.slide_id}_occlusion_heatmap_mean.npz"
        res = evaluate_occlusion_localization_one_slide(item, npz_path)
        if res is not None:
            per_slide.append(res)

    if len(per_slide) == 0:
        return {
            "n_localization_slides": 0,
            "hit_at_1": float("nan"),
            "hit_at_5": float("nan"),
        }

    return {
        "n_localization_slides": int(len(per_slide)),
        "hit_at_1": float(np.mean([r["hit_at_1"] for r in per_slide])),
        "hit_at_5": float(np.mean([r["hit_at_5"] for r in per_slide])),
    }


def make_occlusion_heatmap_one_slide_repeated(
    item: "SlideItem",
    model: nn.Module,
    head: nn.Module,
    device: torch.device,
    k_max: int,
    tile_size: int,
    block_px: int,
    refill: bool,
    seed: int,
    out_dir: Path,
    n_repeats: int,
    thumb_max_px: int = 2048,
    roi_png_dir: Optional[Path] = None,
    pred_label: Optional[int] = None,
    threshold: float = 0.5,
) -> Optional[Dict[str, float]]:
    """
    Repeated sampled-subset occlusion heatmap.
    Main score is chosen by slide prediction direction:
      - predicted positive -> mean block probability drop
      - predicted negative -> mean block probability increase
    No hard filtering is applied. Reliability is surfaced by annotating top-scoring
    blocks with "s|n", where s=support and n=mean tile count for that block.
    If the slide has ROI=1 and none of the top-k annotated blocks overlaps the ROI circle,
    one best ROI-overlapping block is additionally annotated in red.
    """
    all_repeat_scores = {}
    all_repeat_raw_prob_drops = {}
    all_repeat_raw_prob_increases = {}
    all_repeat_counts = {}
    prob_bases = []

    if pred_label is not None:
        pred_label = int(pred_label)

    for r in range(int(n_repeats)):
        res = compute_occlusion_blocks_one_repeat(
            item=item,
            model=model,
            head=head,
            device=device,
            k_max=k_max,
            tile_size=tile_size,
            block_px=block_px,
            refill=refill,
            seed=seed,
            repeat_idx=r,
        )
        if res is None:
            continue

        prob_bases.append(float(res["prob_base"]))

        pred_label_this = int(pred_label) if pred_label is not None else int(float(res["prob_base"]) >= float(threshold))
        score_dict = res["block_scores_drop"] if pred_label_this == 1 else res["block_scores_increase"]
        for key, val in score_dict.items():
            all_repeat_scores.setdefault(key, []).append(val)
        for key, val in res["block_drop_prob_raw"].items():
            all_repeat_raw_prob_drops.setdefault(key, []).append(val)
        for key, val in res["block_increase_prob_raw"].items():
            all_repeat_raw_prob_increases.setdefault(key, []).append(val)
        for key, val in res["block_counts"].items():
            all_repeat_counts.setdefault(key, []).append(val)

    if len(all_repeat_scores) == 0:
        return None

    keys = sorted(all_repeat_scores.keys())
    bx_vals = [k[0] for k in keys]
    by_vals = [k[1] for k in keys]
    bx_min, bx_max = min(bx_vals), max(bx_vals)
    by_min, by_max = min(by_vals), max(by_vals)

    grid_w = bx_max - bx_min + 1
    grid_h = by_max - by_min + 1

    mean_score_grid = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    std_score_grid = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    mean_drop_prob_raw_grid = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    mean_increase_prob_raw_grid = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    support_grid = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    mean_count_grid = np.full((grid_h, grid_w), np.nan, dtype=np.float32)

    for (bxi, byi), vals in all_repeat_scores.items():
        gx = bxi - bx_min
        gy = byi - by_min
        vals_np = np.asarray(vals, dtype=np.float32)
        finite = vals_np[np.isfinite(vals_np)]
        if finite.size > 0:
            mean_score_grid[gy, gx] = float(np.mean(finite))
            std_score_grid[gy, gx] = float(np.std(finite))
            support_grid[gy, gx] = float(finite.size)

        raw_vals = np.asarray(all_repeat_raw_prob_drops.get((bxi, byi), []), dtype=np.float32)
        raw_vals = raw_vals[np.isfinite(raw_vals)]
        if raw_vals.size > 0:
            mean_drop_prob_raw_grid[gy, gx] = float(np.mean(raw_vals))

        inc_vals = np.asarray(all_repeat_raw_prob_increases.get((bxi, byi), []), dtype=np.float32)
        inc_vals = inc_vals[np.isfinite(inc_vals)]
        if inc_vals.size > 0:
            mean_increase_prob_raw_grid[gy, gx] = float(np.mean(inc_vals))

        cnts = np.asarray(all_repeat_counts.get((bxi, byi), []), dtype=np.float32)
        cnts = cnts[np.isfinite(cnts)]
        if cnts.size > 0:
            mean_count_grid[gy, gx] = float(np.mean(cnts))

    prob_base_mean = float(np.mean(prob_bases)) if len(prob_bases) else float("nan")
    if pred_label is None:
        pred_label = int(prob_base_mean >= float(threshold)) if np.isfinite(prob_base_mean) else 0
    score_mode = "drop" if int(pred_label) == 1 else "increase"

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{item.slide_id}_occlusion_heatmap_mean.npz"
    np.savez_compressed(
        npz_path,
        slide_id=item.slide_id,
        prob_base_mean=prob_base_mean,
        block_px=int(block_px),
        bx_min=int(bx_min),
        by_min=int(by_min),
        pred_label=int(pred_label),
        score_mode=score_mode,
        mean_score_grid=mean_score_grid,
        std_score_grid=std_score_grid,
        mean_drop_prob_grid=mean_drop_prob_raw_grid,
        std_drop_prob_grid=std_score_grid,
        mean_drop_prob_raw_grid=mean_drop_prob_raw_grid,
        mean_increase_prob_raw_grid=mean_increase_prob_raw_grid,
        support_grid=support_grid,
        mean_count_grid=mean_count_grid,
        n_repeats=int(n_repeats),
    )

    # render mean heatmap png
    png_path = out_dir / f"{item.slide_id}_occlusion_overlay_mean.png"
    support_png_path = out_dir / f"{item.slide_id}_occlusion_support.png"
    scatter_png_path = out_dir / f"{item.slide_id}_block_importance_vs_tilenum.png"

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        # ------------------------------------------------------------------
        # Choose background image: ROI PNG only.
        # For the user's requested confusion-matrix export, slides without ROI
        # PNG/meta are skipped upstream and logged by name.
        # ------------------------------------------------------------------
        use_roi_png = False
        roi_meta = None

        if roi_png_dir is None:
            raise FileNotFoundError("roi_png_dir is required for ROI PNG heatmap rendering.")

        roi_meta = load_roi_meta_for_slide(roi_png_dir, item.slide_id)
        if roi_meta is None:
            raise FileNotFoundError(f"Missing ROI meta JSON for slide: {item.slide_id}")

        roi_png_result = load_roi_png_for_slide(roi_png_dir, item.slide_id, thumb_max_px)
        if roi_png_result is None:
            raise FileNotFoundError(f"Missing ROI PNG for slide: {item.slide_id}")

        thumb_img, _disp_w, _disp_h, thumb_w, thumb_h = roi_png_result
        use_roi_png = True

        # ------------------------------------------------------------------
        # grid_to_thumb: maps block grid positions → thumbnail pixel rects.
        # When using ROI PNG the global level-0 block coords are first
        # converted into ROI space (subtract origin, divide by scale), then
        # scaled by (thumb / roi_dim).
        # When using the full NDPI thumbnail the old direct scale is used.
        # ------------------------------------------------------------------
        if use_roi_png:
            origin_x = float(roi_meta["origin_x"])
            origin_y = float(roi_meta["origin_y"])
            roi_scale = float(roi_meta["scale"])
            roi_w = int(roi_meta["roi_width"])
            roi_h = int(roi_meta["roi_height"])
            sx = thumb_w / roi_w
            sy = thumb_h / roi_h

            def _global_to_thumb_x(gx_px: float) -> int:
                return int(round(((gx_px - origin_x) / roi_scale) * sx))

            def _global_to_thumb_y(gy_px: float) -> int:
                return int(round(((gy_px - origin_y) / roi_scale) * sy))

        else:
            slide_tmp = openslide.OpenSlide(str(item.ndpi_path))
            w0, h0 = slide_tmp.dimensions
            slide_tmp.close()

            def _global_to_thumb_x(gx_px: float) -> int:
                return int(round(gx_px * (thumb_w / w0)))

            def _global_to_thumb_y(gy_px: float) -> int:
                return int(round(gy_px * (thumb_h / h0)))

        # Convert block-level scores from global slide coordinates into
        # thumbnail pixel coordinates so that the heatmap can be displayed
        # on top of the ROI image.
        def grid_to_thumb(grid: np.ndarray) -> np.ndarray:
            hm = np.full((thumb_h, thumb_w), np.nan, dtype=np.float32)
            for _gy in range(grid_h):
                for _gx in range(grid_w):
                    val = grid[_gy, _gx]
                    if not np.isfinite(val):
                        continue
                    bxi = bx_min + _gx
                    byi = by_min + _gy
                    x0_g = bxi * int(block_px)
                    y0_g = byi * int(block_px)
                    x1_g = x0_g + int(block_px)
                    y1_g = y0_g + int(block_px)

                    tx0 = _global_to_thumb_x(x0_g)
                    ty0 = _global_to_thumb_y(y0_g)
                    tx1 = _global_to_thumb_x(x1_g)
                    ty1 = _global_to_thumb_y(y1_g)

                    tx0 = max(0, min(thumb_w, tx0)); tx1 = max(0, min(thumb_w, tx1))
                    ty0 = max(0, min(thumb_h, ty0)); ty1 = max(0, min(thumb_h, ty1))
                    if tx1 <= tx0 or ty1 <= ty0:
                        continue
                    hm[ty0:ty1, tx0:tx1] = val
            return hm

        # main heatmap: color = score, alpha fixed for cleaner overlay
        hm = grid_to_thumb(mean_score_grid)
        finite = hm[np.isfinite(hm)]
        if finite.size > 0:
            lo = float(np.percentile(finite, 5))
            hi = float(np.percentile(finite, 95))
            if hi <= lo:
                hi = lo + 1e-6
            hm_norm = np.clip((hm - lo) / (hi - lo), 0.0, 1.0)
        else:
            lo, hi = 0.0, 1.0
            hm_norm = np.zeros_like(hm, dtype=np.float32)

        alpha_map = np.where(np.isfinite(hm), 0.45, 0.0).astype(np.float32)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        ax.imshow(thumb_img)

        # draw the true ROI circle on thumbnail for QC
        roi_circle = get_circle_roi_px0(item) if int(item.roi) == 1 else None
        if roi_circle is not None:
            rcx_g, rcy_g, rr_g = roi_circle
            # Convert circle centre from global → thumbnail space
            tcx = _global_to_thumb_x(rcx_g)
            tcy = _global_to_thumb_y(rcy_g)
            # Radius: scale by the x-direction thumbnail-to-global ratio
            if use_roi_png:
                tr = (rr_g / roi_scale) * sx
            else:
                tr = rr_g * (thumb_w / w0)

            roi_patch = Circle(
                (tcx, tcy),
                tr,
                fill=False,
                edgecolor="red",
                linewidth=2.0,
            )
            ax.add_patch(roi_patch)

        im = ax.imshow(hm_norm, alpha=alpha_map, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.axis("off")
        ax.set_title(
            f"{item.slide_id} | mean LVI prob={prob_base_mean:.3f} | pred={'pos' if int(pred_label) == 1 else 'neg'} | "
            f"block={block_px}px | repeats={n_repeats}"
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        # Keep normalized color mapping for display, but label the bar with
        # the corresponding raw mean block probability drop values.
        mid = lo + 0.5 * (hi - lo)

        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels([
            f"≤ {lo:.4g}",
            f"{mid:.4g}",
            f"≥ {hi:.4g}",
        ])
        cbar.set_label(f"Mean block probability {score_mode}\n(percentile-normalized for display)")

        # Annotate only the highest-scoring blocks to avoid clutter.
        # Text format: "s|n" where:
        #   s = support (number of repeats where the block had a finite score)
        #   n = mean tile count in that block across observed repeats
        ann_candidates = []
        roi_overlap_candidates = []
        roi_circle = get_circle_roi_px0(item) if int(item.roi) == 1 else None

        for _gy in range(grid_h):
            for _gx in range(grid_w):
                score = float(mean_score_grid[_gy, _gx])
                sup = float(support_grid[_gy, _gx]) if np.isfinite(support_grid[_gy, _gx]) else float("nan")
                cnt = float(mean_count_grid[_gy, _gx]) if np.isfinite(mean_count_grid[_gy, _gx]) else float("nan")
                if not np.isfinite(score):
                    continue

                bxi = bx_min + _gx
                byi = by_min + _gy
                x0_g = bxi * int(block_px)
                y0_g = byi * int(block_px)
                x1_g = x0_g + int(block_px)
                y1_g = y0_g + int(block_px)

                overlaps_roi = False
                if roi_circle is not None:
                    rcx_g, rcy_g, rr_g = roi_circle
                    overlaps_roi = rect_circle_overlap(x0_g, y0_g, x1_g, y1_g, rcx_g, rcy_g, rr_g)

                ann_candidates.append((score, _gx, _gy, sup, cnt, overlaps_roi))
                if overlaps_roi:
                    roi_overlap_candidates.append((score, _gx, _gy, sup, cnt, overlaps_roi))

        ann_candidates.sort(key=lambda t: t[0], reverse=True)
        roi_overlap_candidates.sort(key=lambda t: t[0], reverse=True)

        n_annot = min(10, len(ann_candidates))
        selected = ann_candidates[:n_annot]

        topk_has_roi_overlap = any(t[5] for t in selected)
        if (roi_circle is not None) and (not topk_has_roi_overlap) and len(roi_overlap_candidates) > 0:
            roi_best = roi_overlap_candidates[0]
            already_selected = any((roi_best[1] == t[1] and roi_best[2] == t[2]) for t in selected)
            if not already_selected:
                selected.append(roi_best)

        for _, _gx, _gy, sup, cnt, overlaps_roi in selected:
            bxi = bx_min + _gx
            byi = by_min + _gy
            x0_g = bxi * int(block_px)
            y0_g = byi * int(block_px)
            x1_g = x0_g + int(block_px)
            y1_g = y0_g + int(block_px)

            tx0 = _global_to_thumb_x(x0_g); tx1 = _global_to_thumb_x(x1_g)
            ty0 = _global_to_thumb_y(y0_g); ty1 = _global_to_thumb_y(y1_g)

            tx0 = max(0, min(thumb_w, tx0)); tx1 = max(0, min(thumb_w, tx1))
            ty0 = max(0, min(thumb_h, ty0)); ty1 = max(0, min(thumb_h, ty1))
            if tx1 <= tx0 or ty1 <= ty0:
                continue

            cx_ann = 0.5 * (tx0 + tx1)
            cy_ann = 0.5 * (ty0 + ty1)
            label = f"{int(round(sup))}|{cnt:.1f}"
            text_color = "red" if overlaps_roi and (roi_circle is not None) else "white"
            ax.text(
                cx_ann, cy_ann, label,
                ha="center", va="center",
                fontsize=7, color=text_color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.55, edgecolor="none"),
            )

        ax.text(
            0.01, 0.01,
            "block label: s|n  (s = support, n = mean tile count; red = ROI)",
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.60, edgecolor="none"),
        )

        plt.tight_layout()
        plt.savefig(png_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        # support map (kept as auxiliary QC image)
        hm_sup = grid_to_thumb(support_grid.astype(np.float32))
        finite_sup = hm_sup[np.isfinite(hm_sup)]
        if finite_sup.size > 0:
            hi_sup = float(np.max(finite_sup))
            hm_sup_norm = hm_sup / max(hi_sup, 1.0)
        else:
            hm_sup_norm = np.zeros_like(hm_sup, dtype=np.float32)

        plt.figure(figsize=(8, 8), dpi=150)
        plt.imshow(thumb_img)
        plt.imshow(hm_sup_norm, alpha=0.45, cmap="viridis")
        plt.axis("off")
        plt.title(f"{item.slide_id} | support map | repeats={n_repeats}")
        plt.tight_layout()
        plt.savefig(support_png_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()

        # scatter plot: x = mean tile count per block,
        # y = mean block importance using the same rule as the main heatmap
        # (predicted positive -> drop, predicted negative -> increase)
        x_vals = mean_count_grid.reshape(-1)
        y_vals = mean_score_grid.reshape(-1)

        finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_scatter = x_vals[finite_mask]
        y_scatter = y_vals[finite_mask]

        plt.figure(figsize=(6, 6), dpi=150)
        plt.scatter(x_scatter, y_scatter, s=18, alpha=0.7)
        plt.axhline(0.0, linewidth=1.0, alpha=0.8)
        plt.xlabel("Block tile number")
        plt.ylabel(f"Importance (probability {score_mode} per block)")
        plt.title(f"{item.slide_id}\nimportance vs tile number per block")
        plt.tight_layout()
        plt.savefig(scatter_png_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()
    except Exception as e:
        err_path = out_dir / f"{item.slide_id}_occlusion_error.txt"
        err_path.write_text(str(e), encoding="utf-8")

    return {
        "slide_id": item.slide_id,
        "prob_base_mean": float(prob_base_mean),
        "pred_label": int(pred_label),
        "score_mode": score_mode,
        "npz": str(npz_path),
        "png": str(png_path),
        "support_png": str(support_png_path),
        "scatter_png": str(scatter_png_path),
    }

def train_one_fold(
    args,
    train_items: List[SlideItem],
    val_items: List[SlideItem],
    test_items: Optional[List[SlideItem]],
    out_dir: Path,
) -> Dict[str, float]:
    """Train on train_items, early-stop on val_items, optionally evaluate on test_items."""

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Model: frozen slide encoder + trainable head
    model = load_slide_encoder(global_pool=True).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    emb_dim = infer_embed_dim(model, device)
    head = nn.Linear(emb_dim, 1).to(device)

    opt = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr_head,
        weight_decay=args.weight_decay,
    )

    # BCE (pos_weight optional)
    # If --use_pos_weight is enabled, positive slides receive higher weight
    # according to the negative/positive ratio in the training set.
    # This can help when LVI-positive slides are underrepresented.
    pos = sum(it.y for it in train_items)
    neg = len(train_items) - pos
    if args.use_pos_weight and pos > 0:
        pos_weight = torch.tensor([neg / max(pos, 1)], device=device, dtype=torch.float32)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using pos_weight={pos_weight.item():.4f}")
    else:
        bce = nn.BCEWithLogitsLoss()

    roi_cache = RoiRadiusCache()

    best_val_auc = -1.0
    best_path = out_dir / "best.pt"
    no_improve = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Main training loop.
    # Each epoch samples a new subset of tiles from every slide.
    for epoch in range(1, args.epochs + 1):
        model.eval()
        head.train()

        # reproducible-but-changing sampling per epoch
        rng = np.random.RandomState(args.seed + epoch)
        random.shuffle(train_items)

        train_losses = []
        train_probs = []
        train_true = []

        opt.zero_grad(set_to_none=True)

        for i, item in enumerate(train_items):
            tile_embeds_np, coords_np, dist_np = read_h5_tile_data(item.h5_path)
            n_tiles = tile_embeds_np.shape[0]
            if n_tiles == 0:
                continue

            # ROI-priority sampling ONLY for positive slides with ROI flag.
            # Strategy:
            #   - strictly define inside ROI as tile-centre distance <= ROI radius
            #   - take up to args.roi_inside_cap tiles from inside ROI
            #   - fill the remaining budget from global random tiles
            use_roi_sampling = (item.y == 1) and (item.roi == 1)
            roi_radius_px0 = roi_cache.get_roi_radius_px0(item) if use_roi_sampling else None

            idx = sample_tile_indices(
                dist_to_roi=dist_np,
                n_tiles=n_tiles,
                k_max=args.k_max,
                tile_size=args.tile_size,
                margin_px=args.margin_px,
                roi_radius_px0=roi_radius_px0,
                use_roi_sampling=use_roi_sampling,
                roi_frac=args.roi_frac,
                rng=rng,
                roi_sampling_mode="inside_priority" if use_roi_sampling else "near",
                roi_inside_cap=args.roi_inside_cap,
            )

            emb = torch.from_numpy(tile_embeds_np[idx]).to(
                device=device,
                dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            coords = torch.from_numpy(coords_np[idx]).to(device=device, dtype=torch.float32)

            emb = emb.unsqueeze(0)
            coords = coords.unsqueeze(0)
            y = torch.tensor([item.y], device=device, dtype=torch.float32)

            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                slide_vec = forward_slide_encoder(model, emb, coords)
                logit = head(slide_vec).squeeze(1)
                loss = bce(logit, y) / args.grad_accum

            scaler.scale(loss).backward()

            if ((i + 1) % args.grad_accum == 0) or (i == len(train_items) - 1):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            train_losses.append(float(loss.item() * args.grad_accum))
            prob = float(torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)[0])
            train_probs.append(prob)
            train_true.append(item.y)

        train_loss = float(np.mean(train_losses)) if len(train_losses) else float("nan")
        train_auc = compute_auc(np.asarray(train_true, np.int64), np.asarray(train_probs, np.float32))

        # Validation (uniform sampling; repeated)
        val_metrics = evaluate(
            val_items,
            model,
            head,
            device,
            k_max=args.k_max,
            tile_size=args.tile_size,
            margin_px=args.margin_px,
            roi_frac=args.roi_frac,
            seed=args.seed,
            n_samples=args.eval_samples_val,
        )

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_auc={train_auc:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} val_auc={val_metrics['val_auc']:.4f}"
        )
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("auc/train", train_auc, epoch)
        writer.add_scalar("loss/val", val_metrics["val_loss"], epoch)
        writer.add_scalar("auc/val", val_metrics["val_auc"], epoch)

        val_auc = val_metrics["val_auc"]
        # Save a new best checkpoint only when validation AUC improves.
        # If validation AUC does not improve for args.patience epochs,
        # training stops early to reduce overfitting.
        if np.isfinite(val_auc) and (val_auc > best_val_auc + 1e-4):
            best_val_auc = float(val_auc)
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "slide_encoder_state": model.state_dict(),
                    "head_state": head.state_dict(),
                    "best_val_auc": best_val_auc,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  [SAVE] best.pt (val_auc={best_val_auc:.4f})")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(
                f"[EARLY STOP] no val AUC improvement for {args.patience} epochs. "
                f"Best val_auc={best_val_auc:.4f}"
            )
            break

    writer.close()

    # Load best for final eval
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["slide_encoder_state"])
        head.load_state_dict(ckpt["head_state"])

    out = {"best_val_auc": float(best_val_auc)}
    
    # Re-evaluate the selected best model on val to choose a classification threshold.
    val_metrics_best = evaluate(
        val_items,
        model,
        head,
        device,
        k_max=args.k_max,
        tile_size=args.tile_size,
        margin_px=args.margin_px,
        roi_frac=args.roi_frac,
        seed=args.seed,
        n_samples=args.eval_samples_val,
    )
    
    thr_sel = select_threshold_for_target_sensitivity(
        y_true=np.asarray(val_metrics_best["y_true"], dtype=np.int64),
        y_prob=np.asarray(val_metrics_best["y_prob"], dtype=np.float32),
        target_sensitivity=0.75,
    )
    out["selected_threshold"] = float(thr_sel["selected_threshold"])
    out["selected_val_sensitivity"] = float(thr_sel["selected_val_sensitivity"])
    out["selected_val_specificity"] = float(thr_sel["selected_val_specificity"])
    
    if test_items is not None:
        test_metrics = evaluate(
            test_items,
            model,
            head,
            device,
            k_max=args.k_max,
            tile_size=args.tile_size,
            margin_px=args.margin_px,
            roi_frac=args.roi_frac,
            seed=args.seed + 12345,
            n_samples=args.eval_samples_test,
        )
        out["test_loss"] = float(test_metrics["val_loss"])
        out["test_auc"] = float(test_metrics["val_auc"])
        out["test_y_true"] = test_metrics["y_true"]
        out["test_y_prob"] = test_metrics["y_prob"]
    
        test_thr_metrics = sensitivity_specificity_from_threshold(
            y_true=np.asarray(test_metrics["y_true"], dtype=np.int64),
            y_prob=np.asarray(test_metrics["y_prob"], dtype=np.float32),
            threshold=float(out["selected_threshold"]),
        )
        out["test_sensitivity"] = float(test_thr_metrics["sensitivity"])
        out["test_specificity"] = float(test_thr_metrics["specificity"])
    
        print(
            f"[TEST] loss={out['test_loss']:.4f} auc={out['test_auc']:.4f} | "
            f"thr={out['selected_threshold']:.4f} "
            f"val_sens={out['selected_val_sensitivity']:.4f} "
            f"val_spec={out['selected_val_specificity']:.4f} | "
            f"test_sens={out['test_sensitivity']:.4f} "
            f"test_spec={out['test_specificity']:.4f}"
        )
        # Optional: occlusion/ablation heatmaps on held-out data
        if args.make_heatmaps:
            split_items = test_items if args.heatmap_split == "test" else val_items
            hm_summary = generate_confusion_heatmaps_for_split(
                split_items=split_items,
                model=model,
                head=head,
                device=device,
                threshold=float(out["selected_threshold"]),
                args=args,
                out_dir=out_dir,
            )
            out["heatmap_threshold"] = float(hm_summary["threshold"])
            out["heatmap_split_used"] = hm_summary["split"]
            out["heatmap_available_roi_png_n"] = int(len(hm_summary["available_roi_png_slides"]))
            out["heatmap_missing_roi_png_n"] = int(len(hm_summary["missing_roi_png_slides"]))
            out["heatmap_generated_n"] = int(len(hm_summary["generated_heatmaps"]))
            out["heatmap_selected_counts"] = {
                k: int(len(v)) for k, v in hm_summary["selected_examples"].items()
            }
        with open(out_dir / "fold_metrics.json", "w") as f:
            json.dump(out, f, indent=2)
    
        return out
    

# -----------------------------
# optional Single split (without CV) 
# -----------------------------

def run_single_split(args):
    set_seed(args.seed)

    items = build_items(args)

    y_all = np.array([it.y for it in items], dtype=np.int64)
    train_items, tmp_items = train_test_split(
        items,
        test_size=(args.val_frac + args.test_frac),
        random_state=args.seed,
        stratify=y_all,
    )

    y_tmp = np.array([it.y for it in tmp_items], dtype=np.int64)
    val_rel = args.val_frac / (args.val_frac + args.test_frac)
    val_items, test_items = train_test_split(
        tmp_items,
        test_size=(1 - val_rel),
        random_state=args.seed,
        stratify=y_tmp,
    )

    print(f"Split: train={len(train_items)} val={len(val_items)} test={len(test_items)}")
    print(
        f"Train positives={sum(it.y for it in train_items)}  "
        f"Val positives={sum(it.y for it in val_items)}  "
        f"Test positives={sum(it.y for it in test_items)}"
    )

    out_dir = Path(args.out_dir)
    metrics = train_one_fold(args, train_items, val_items, test_items, out_dir)

    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


# -----------------------------
# K-fold CV
# -----------------------------

def run_cv(args):
    set_seed(args.seed)

    items = build_items(args)
    y_all = np.array([it.y for it in items], dtype=np.int64)

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    fold_results = []

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(items)), y_all), start=1):
        trainval_items = [items[i] for i in trainval_idx]
        test_items = [items[i] for i in test_idx]

        y_trainval = np.array([it.y for it in trainval_items], dtype=np.int64)
        train_items, val_items = train_test_split(
            trainval_items,
            test_size=args.cv_val_frac,
            random_state=args.seed + fold,
            stratify=y_trainval,
        )

        fold_dir = out_root / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n===== [CV] Fold {fold}/{args.cv_folds} =====")
        print(f"train={len(train_items)} val={len(val_items)} test={len(test_items)}")
        print(
            f"pos(train)={sum(it.y for it in train_items)}  "
            f"pos(val)={sum(it.y for it in val_items)}  "
            f"pos(test)={sum(it.y for it in test_items)}"
        )

        # per-fold reproducibility (different, but deterministic)
        args_fold = argparse.Namespace(**vars(args))
        args_fold.seed = int(args.seed + fold * 1000)

        m = train_one_fold(args_fold, train_items, val_items, test_items, fold_dir)
        m["fold"] = fold
        m["n_train"] = len(train_items)
        m["n_val"] = len(val_items)
        m["n_test"] = len(test_items)
        fold_results.append(m)

    test_aucs = [m.get("test_auc", float("nan")) for m in fold_results]
    finite_aucs = [a for a in test_aucs if np.isfinite(a)]
    best_val_aucs = [m.get("best_val_auc", float("nan")) for m in fold_results]
    finite_best_val_aucs = [a for a in best_val_aucs if np.isfinite(a)]

    test_sens = [m.get("test_sensitivity", float("nan")) for m in fold_results]
    finite_test_sens = [x for x in test_sens if np.isfinite(x)]
    test_specs = [m.get("test_specificity", float("nan")) for m in fold_results]
    finite_test_specs = [x for x in test_specs if np.isfinite(x)]
    
    summary = {
        "cv_folds": int(args.cv_folds),
        "cv_val_frac": float(args.cv_val_frac),
        "mean_test_auc": float(np.mean(finite_aucs)) if len(finite_aucs) else float("nan"),
        "std_test_auc": float(np.std(finite_aucs)) if len(finite_aucs) else float("nan"),
    
        "mean_best_val_auc": float(np.mean(finite_best_val_aucs)) if len(finite_best_val_aucs) else float("nan"),
        "std_best_val_auc": float(np.std(finite_best_val_aucs)) if len(finite_best_val_aucs) else float("nan"),
    
        "mean_test_sensitivity": float(np.mean(finite_test_sens)) if len(finite_test_sens) else float("nan"),
        "std_test_sensitivity": float(np.std(finite_test_sens)) if len(finite_test_sens) else float("nan"),
        "mean_test_specificity": float(np.mean(finite_test_specs)) if len(finite_test_specs) else float("nan"),
        "std_test_specificity": float(np.std(finite_test_specs)) if len(finite_test_specs) else float("nan"),
    
        "folds": fold_results,
    }

    roc_summary = average_roc_curves_across_folds(fold_results, num_points=101)
    if roc_summary is not None:
        summary["mean_roc_curve"] = roc_summary
        with open(out_root / "mean_roc_curve.json", "w") as f:
            json.dump(roc_summary, f, indent=2)
        plot_mean_roc_curve(roc_summary, out_root / "mean_roc_curve.png", mean_auc=summary["mean_test_auc"], std_auc=summary["std_test_auc"])

    with open(out_root / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== [CV SUMMARY] =====")
    print(f"mean_test_auc={summary['mean_test_auc']:.4f}  std={summary['std_test_auc']:.4f}")
    print(f"mean_best_val_auc={summary['mean_best_val_auc']:.4f}  std={summary['std_best_val_auc']:.4f}")
    print(f"mean_test_sensitivity={summary['mean_test_sensitivity']:.4f}  std={summary['std_test_sensitivity']:.4f}")
    print(f"mean_test_specificity={summary['mean_test_specificity']:.4f}  std={summary['std_test_specificity']:.4f}")


# -----------------------------
# CLI
# -----------------------------

def build_argparser():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--slides_dir",
        type=str,
        required=True,
        help="Folder containing *.ndpi and *.ndpi.ndpa (e.g., /mnt/d/BMM_LVI)",
    )
    ap.add_argument(
        "--h5_root",
        type=str,
        required=True,
        help="Folder containing *.tile_embeds.h5 (e.g., /mnt/d/tile_encoder_h5files)",
    )
    ap.add_argument(
        "--labels_tsv",
        type=str,
        required=True,
        help="TSV with columns: id ROI LVI (id is slide_id like 00PH05780.ndpi)",
    )

    ap.add_argument(
        "--out_dir",
        type=str,
        default="./runs/lvi_gigapath",
        help="Output directory for logs/checkpoints",
    )

    # Single split
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)

    # CV
    ap.add_argument(
        "--cv_folds",
        type=int,
        default=0,
        help="If >0, run stratified K-fold CV with this many folds. If 0, run single split.",
    )
    ap.add_argument(
        "--cv_val_frac",
        type=float,
        default=0.15,
        help="Within each CV fold, fraction of trainval used as val for early stopping.",
    )

    # Sampling / oversampling
    ap.add_argument("--k_max", type=int, default=512, help="Max tiles per slide per step")
    ap.add_argument("--tile_size", type=int, default=1024, help="Tile size used in tiling (geometry)")
    ap.add_argument("--margin_px", type=float, default=1024.0, help="Margin (level-0 px) around ROI")
    ap.add_argument(
        "--roi_frac",
        type=float,
        default=0.5,
        help="Legacy near-ROI sampling fraction. Kept for compatibility; not used by the training-time inside-ROI priority sampler.",
    )
    ap.add_argument(
        "--roi_inside_cap",
        type=int,
        default=256,
        help="For training on y=1 & roi=1 slides: sample up to this many tiles whose centres fall inside the ROI, then fill the rest globally to k_max.",
    )

    # Occlusion heatmaps (feature importance)
    ap.add_argument("--make_heatmaps", action="store_true",
                    help="If set, compute occlusion/ablation heatmaps for TP/TN/FP/FN examples selected using the same threshold as sensitivity/specificity.")
    ap.add_argument("--heatmap_split", type=str, default="test", choices=["test", "val"],
                    help="Which split to generate heatmaps for (per fold).")
    ap.add_argument("--heatmap_block_px", type=int, default=4096,
                    help="Block size in level-0 pixels for occlusion grouping (larger = faster/coarser).")
    ap.add_argument("--heatmap_refill", action="store_true",
                    help="If set, refill occluded tiles with tiles sampled from outside the removed block to keep K constant.")
    ap.add_argument("--heatmap_thumb_max_px", type=int, default=2048,
                    help="Max thumbnail side length (px) for heatmap overlay rendering.")
    ap.add_argument("--roi_png_dir", type=str, default=None,
                    help="Directory containing {slide_id}_roi (PNG) and {slide_id}_roi_meta (JSON) files "
                         "(e.g. /mnt/d/cutting_figures). When provided, heatmaps are rendered on the "
                         "ROI-cropped image instead of the full NDPI thumbnail.")
    ap.add_argument("--heatmap_repeats", type=int, default=30, help="Number of repeated subset samples per slide when building mean occlusion heatmaps.")
    # Eval stability
    ap.add_argument("--eval_samples_val", type=int, default=10, help="Repeated samples per slide in validation")
    ap.add_argument("--eval_samples_test", type=int, default=30, help="Repeated samples per slide in testing")

    # Optimization
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")

    # (kept for compatibility; slide encoder is frozen in this script)
    ap.add_argument("--lr_slide", type=float, default=1e-5)

    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--use_pos_weight", action="store_true", help="Use pos_weight in BCE")

    ap.add_argument("--seed", type=int, default=42)

    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()

    if args.cv_folds and args.cv_folds > 0:
        run_cv(args)
    else:
        run_single_split(args)
