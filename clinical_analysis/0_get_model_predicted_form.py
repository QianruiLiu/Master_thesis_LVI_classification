#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
usage:
python 0_get_model_predicted_form.py   --run_dir "/mnt/d/runs/final_external_eval/lr3e-3_wd1e-2_k512_seed77"   --external_tsv "/mnt/d/labels_external.tsv"   --h5_root "/mnt/d/tile_encoder_h5files"   --slide_patient_map "/mnt/d/slide_vs_patientid.tsv"   --out_dir "/mnt/d/runs/final_external_eval/lr3e-3_wd1e-2_k512_seed77"
Export external-test slide predictions from an already trained GigaPath LVI model.

What this script does:
1) Loads best.pt and final_metrics.json from an existing run directory
2) Reads selected_threshold from final_metrics.json
3) Rebuilds the external set from external_tsv
4) Re-runs inference on the external set (NO retraining)
5) Merges slide_id -> patient_id from a user-provided mapping table
6) Exports a TSV:
      patient_id, slide_id, y_true, model_score, model_group

Important:
- selected_threshold is NOT written as a column in the TSV.
- selected_threshold IS embedded in the output filename.
- model_group is:
      high  if model_score >= selected_threshold
      low   otherwise

Compatible with the training script structure where:
- best.pt stores:
    - slide_encoder_state
    - head_state
    - args
- final_metrics.json stores:
    - selected_threshold
- external_tsv has columns:
    - id, ROI, LVI
"""

import argparse
import json
import math
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import gigapath.slide_encoder as slide_encoder


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def _stable_int_from_str(s: str) -> int:
    return int(zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF)


def infer_sep(path: Path) -> Optional[str]:
    """Infer separator from suffix, else return None for pandas auto-detect."""
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return "\t"
    if suffix == ".csv":
        return ","
    return None


def read_table_auto(path: Path) -> pd.DataFrame:
    sep = infer_sep(path)
    if sep is not None:
        return pd.read_csv(path, sep=sep)
    return pd.read_csv(path, sep=None, engine="python")


def format_threshold_for_filename(thr: float) -> str:
    """
    Convert 0.1984 -> '0p1984'
    Keeps a compact but readable filename-safe representation.
    """
    s = f"{float(thr):.4f}"
    return s.replace(".", "p").replace("-", "m")


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class SlideItem:
    slide_id: str
    h5_path: Path
    y: int


# -----------------------------
# Model / feature loading
# -----------------------------

def read_h5_tile_data(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (tile_embeds, coords, dist_to_roi).

    tile_embeds: [N, 1536]
    coords:      [N, 2]
    dist_to_roi: [N], NaN if not available
    """
    with h5py.File(h5_path, "r") as f:
        tile_embeds = f["tile_embeds"][:]
        coords = f["coords"][:]
        dist_to_roi = (
            f["dist_to_roi"][:] if "dist_to_roi" in f
            else np.full((coords.shape[0],), np.nan, np.float32)
        )
    return tile_embeds, coords, dist_to_roi


def load_slide_encoder(global_pool: bool = True) -> nn.Module:
    model = slide_encoder.create_model(
        "hf_hub:prov-gigapath/prov-gigapath",
        "gigapath_slide_enc12l768d",
        1536,
        global_pool=global_pool,
    )
    return model


def forward_slide_encoder(model: nn.Module, tile_embeds: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    out = model(tile_embeds, coords, all_layer_embed=False)
    if isinstance(out, (list, tuple)):
        out = out[-1]
    elif isinstance(out, dict):
        out = out.get("last_layer_embed", list(out.values())[-1])
    return out


@torch.no_grad()
def predict_logit_for_indices(
    model: nn.Module,
    head: nn.Module,
    tile_embeds_np: np.ndarray,
    coords_np: np.ndarray,
    idx: np.ndarray,
    device: torch.device,
) -> float:
    if idx.size == 0:
        return float("nan")

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

    return float(logit.detach().cpu().numpy().reshape(-1)[0])


# -----------------------------
# Sampling / prediction
# -----------------------------

def sample_tile_indices_uniform(
    n_tiles: int,
    k_max: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Uniform sampling only.
    This matches val/test evaluation behavior in your training script:
    no ROI oversampling for evaluation.
    """
    k_total = min(int(k_max), int(n_tiles))
    all_idx = np.arange(n_tiles, dtype=np.int64)
    if n_tiles <= k_total:
        return all_idx
    return rng.choice(all_idx, size=k_total, replace=False)


@torch.no_grad()
def compute_slide_mean_probability(
    item: SlideItem,
    model: nn.Module,
    head: nn.Module,
    device: torch.device,
    k_max: int,
    seed: int,
    n_samples: int,
) -> float:
    """
    Repeated uniform-sampling prediction for one slide.
    Returns mean LVI probability across repeats.
    """
    tile_embeds_np, coords_np, _dist_np = read_h5_tile_data(item.h5_path)
    n_tiles = tile_embeds_np.shape[0]
    if n_tiles == 0:
        return float("nan")

    base_seed = int(seed) + _stable_int_from_str(item.slide_id) % 1000003
    probs: List[float] = []

    for r in range(int(n_samples)):
        rng = np.random.RandomState(base_seed + int(r) * 10007)
        idx = sample_tile_indices_uniform(
            n_tiles=n_tiles,
            k_max=k_max,
            rng=rng,
        )
        logit = predict_logit_for_indices(model, head, tile_embeds_np, coords_np, idx, device)
        if np.isfinite(logit):
            probs.append(float(1.0 / (1.0 + np.exp(-logit))))

    if len(probs) == 0:
        return float("nan")
    return float(np.mean(np.asarray(probs, dtype=np.float32)))


# -----------------------------
# Data building
# -----------------------------

def build_items_from_tsv(labels_tsv: Path, h5_root: Path) -> List[SlideItem]:
    """
    Build external SlideItem list from external_tsv and h5_root.
    Requires columns in labels_tsv:
      - id
      - ROI
      - LVI
    """
    lab = pd.read_csv(labels_tsv, sep="\t")
    lab.columns = [c.strip() for c in lab.columns]

    required = {"id", "ROI", "LVI"}
    missing = required - set(lab.columns)
    if missing:
        raise ValueError(f"{labels_tsv} is missing required columns: {sorted(missing)}")

    lab["ROI"] = lab["ROI"].apply(safe_int01)
    lab["LVI"] = lab["LVI"].apply(safe_int01)

    h5_files = sorted(h5_root.glob("*.tile_embeds.h5"))
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No H5 files found in: {h5_root}")

    h5_map = {p.name.replace(".tile_embeds.h5", ""): p for p in h5_files}

    items: List[SlideItem] = []
    missing_h5: List[str] = []

    # Keep the external_tsv order
    for _, row in lab.iterrows():
        slide_id = str(row["id"]).strip()
        if slide_id not in h5_map:
            missing_h5.append(slide_id)
            continue

        items.append(
            SlideItem(
                slide_id=slide_id,
                h5_path=h5_map[slide_id],
                y=int(row["LVI"]),
            )
        )

    print(f"[INFO] external_tsv rows: {len(lab)}")
    print(f"[INFO] matched external slides with H5: {len(items)}")
    if missing_h5:
        print(f"[WARN] {len(missing_h5)} external slides missing H5 features.")
        print(f"[WARN] First few missing H5 slide_ids: {missing_h5[:10]}")

    return items


def load_slide_patient_map(
    map_path: Path,
    slide_col: str,
    patient_col: str,
) -> pd.DataFrame:
    """
    Load slide_id -> patient_id mapping table.
    The user can specify the source column names via CLI.
    """
    df = read_table_auto(map_path)
    df.columns = [str(c).strip() for c in df.columns]

    if slide_col not in df.columns:
        raise ValueError(f"Mapping file missing slide column: {slide_col}. Available: {df.columns.tolist()}")
    if patient_col not in df.columns:
        raise ValueError(f"Mapping file missing patient column: {patient_col}. Available: {df.columns.tolist()}")

    out = df[[slide_col, patient_col]].copy()
    out.columns = ["slide_id", "patient_id"]

    out["slide_id"] = out["slide_id"].astype(str).str.strip()
    out["patient_id"] = out["patient_id"].astype(str).str.strip()

    out = out.dropna(subset=["slide_id", "patient_id"])
    out = out[out["slide_id"] != ""]
    out = out[out["patient_id"] != ""]

    # Check duplicates
    dup_slide = out["slide_id"].duplicated(keep=False)
    if dup_slide.any():
        bad = out.loc[dup_slide].sort_values("slide_id")
        # If same slide maps to multiple patients, fail loudly.
        nunique = bad.groupby("slide_id")["patient_id"].nunique()
        truly_bad = nunique[nunique > 1]
        if len(truly_bad) > 0:
            raise ValueError(
                "Some slide_id values map to multiple patient_id values in the mapping file. "
                f"Examples: {truly_bad.index.tolist()[:10]}"
            )
        # Otherwise collapse exact duplicates.
        out = out.drop_duplicates(subset=["slide_id", "patient_id"])

    out = out.drop_duplicates(subset=["slide_id"]).reset_index(drop=True)
    print(f"[INFO] loaded slide->patient mapping rows: {len(out)}")
    return out


# -----------------------------
# Checkpoint / config loading
# -----------------------------

def load_selected_threshold(final_metrics_path: Path) -> float:
    with open(final_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    if "selected_threshold" not in metrics:
        raise KeyError(f"'selected_threshold' not found in {final_metrics_path}")

    thr = float(metrics["selected_threshold"])
    if not np.isfinite(thr):
        raise ValueError(f"selected_threshold is not finite in {final_metrics_path}: {thr}")

    return thr


def load_checkpoint_and_args(best_pt_path: Path, device: torch.device) -> Tuple[dict, Dict]:
    ckpt = torch.load(best_pt_path, map_location=device)
    if "head_state" not in ckpt or "slide_encoder_state" not in ckpt:
        raise KeyError(f"{best_pt_path} does not contain required keys: head_state / slide_encoder_state")

    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}

    return ckpt, ckpt_args


def resolve_param(cli_value, ckpt_args: Dict, key: str, default=None):
    if cli_value is not None:
        return cli_value
    if key in ckpt_args:
        return ckpt_args[key]
    return default


# -----------------------------
# Main export
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export external predictions from an already trained GigaPath LVI model."
    )
    parser.add_argument("--run_dir", required=True, help="Run directory containing best.pt and final_metrics.json")
    parser.add_argument("--external_tsv", required=True, help="External TSV used for test/evaluation")
    parser.add_argument("--h5_root", required=True, help="Directory containing *.tile_embeds.h5 files")
    parser.add_argument("--slide_patient_map", required=True, help="Table mapping slide_id -> patient_id")
    parser.add_argument("--out_dir", required=True, help="Output directory for exported TSV")

    parser.add_argument("--map_slide_col", default="slide_id", help="Column name for slide ID in mapping table")
    parser.add_argument("--map_patient_col", default="patient_id", help="Column name for patient ID in mapping table")

    # Optional overrides; if omitted, values are taken from best.pt args when available
    parser.add_argument("--k_max", type=int, default=None, help="Override k_max")
    parser.add_argument("--eval_samples_test", type=int, default=None, help="Override eval_samples_test")
    parser.add_argument("--seed", type=int, default=None, help="Override seed used for evaluation")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    external_tsv = Path(args.external_tsv)
    h5_root = Path(args.h5_root)
    map_path = Path(args.slide_patient_map)
    out_dir = Path(args.out_dir)

    best_pt = run_dir / "best.pt"
    final_metrics = run_dir / "final_metrics.json"

    if not best_pt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {best_pt}")
    if not final_metrics.exists():
        raise FileNotFoundError(f"Missing final_metrics.json: {final_metrics}")
    if not external_tsv.exists():
        raise FileNotFoundError(f"Missing external_tsv: {external_tsv}")
    if not h5_root.exists():
        raise FileNotFoundError(f"Missing h5_root: {h5_root}")
    if not map_path.exists():
        raise FileNotFoundError(f"Missing slide_patient_map: {map_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load threshold from final_metrics.json
    selected_threshold = load_selected_threshold(final_metrics)
    thr_tag = format_threshold_for_filename(selected_threshold)

    # Load checkpoint
    ckpt, ckpt_args = load_checkpoint_and_args(best_pt, device)

    # Resolve evaluation params: CLI override > checkpoint args > fallback
    k_max = int(resolve_param(args.k_max, ckpt_args, "k_max", 512))
    eval_samples_test = int(resolve_param(args.eval_samples_test, ckpt_args, "eval_samples_test", 30))
    seed = int(resolve_param(args.seed, ckpt_args, "seed", 42))

    print(f"[INFO] selected_threshold = {selected_threshold:.6f}")
    print(f"[INFO] k_max = {k_max}")
    print(f"[INFO] eval_samples_test = {eval_samples_test}")
    print(f"[INFO] seed = {seed}")
    print(f"[INFO] device = {device}")

    set_seed(seed)

    # Build external items
    external_items = build_items_from_tsv(external_tsv, h5_root)
    if len(external_items) == 0:
        raise RuntimeError("No external items could be built.")

    # Build model and head
    model = load_slide_encoder(global_pool=True).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    head_weight = ckpt["head_state"].get("weight", None)
    if head_weight is None:
        raise KeyError("Checkpoint head_state missing 'weight'")
    head_in_features = int(head_weight.shape[1])

    head = nn.Linear(head_in_features, 1).to(device)
    head.eval()

    model.load_state_dict(ckpt["slide_encoder_state"])
    head.load_state_dict(ckpt["head_state"])

    # Predict
    rows = []
    for i, item in enumerate(external_items, start=1):
        prob = compute_slide_mean_probability(
            item=item,
            model=model,
            head=head,
            device=device,
            k_max=k_max,
            seed=seed + 12345,   # matches test-time offset convention in the training script
            n_samples=eval_samples_test,
        )

        if not np.isfinite(prob):
            print(f"[WARN] invalid probability for slide {item.slide_id}; skipping")
            continue

        rows.append({
            "slide_id": item.slide_id,
            "y_true": int(item.y),
            "model_score": float(prob),
        })

        if (i % 20 == 0) or (i == len(external_items)):
            print(f"[INFO] predicted {i}/{len(external_items)} slides")

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        raise RuntimeError("No valid predictions were produced.")

    # Load and merge patient mapping
    map_df = load_slide_patient_map(
        map_path=map_path,
        slide_col=args.map_slide_col,
        patient_col=args.map_patient_col,
    )

    merged = pred_df.merge(map_df, on="slide_id", how="left", validate="one_to_one")

    missing_patient = merged["patient_id"].isna()
    if missing_patient.any():
        bad_slides = merged.loc[missing_patient, "slide_id"].tolist()
        raise ValueError(
            "Some predicted slides could not be matched to patient_id. "
            f"Count={len(bad_slides)}. First few slide_ids: {bad_slides[:10]}"
        )

    # Reorder columns as requested
    merged = merged[["patient_id", "slide_id", "y_true", "model_score"]].copy()

    # Stable sort for readability
    merged = merged.sort_values(["patient_id", "slide_id"]).reset_index(drop=True)

    out_path = out_dir / f"external_predictions_selectedthr{thr_tag}.tsv"
    merged.to_csv(out_path, sep="\t", index=False)

    # Also export a tiny metadata JSON for traceability
    meta = {
        "run_dir": str(run_dir),
        "best_pt": str(best_pt),
        "final_metrics_json": str(final_metrics),
        "external_tsv": str(external_tsv),
        "slide_patient_map": str(map_path),
        "selected_threshold": float(selected_threshold),
        "k_max": int(k_max),
        "eval_samples_test": int(eval_samples_test),
        "seed_used_for_prediction": int(seed + 12345),
        "n_exported_rows": int(len(merged)),
        "output_tsv": str(out_path),
    }
    meta_path = out_dir / f"external_predictions_selectedthr{thr_tag}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] wrote: {out_path}")
    print(f"[DONE] wrote: {meta_path}")


if __name__ == "__main__":
    main()