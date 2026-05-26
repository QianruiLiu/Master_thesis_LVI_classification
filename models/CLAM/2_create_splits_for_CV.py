#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script creates stratified 5-fold cross-validation split files for the CLAM
baseline experiment. It uses the same general development-stage splitting logic
as the GigaPath model comparison: each outer fold has a held-out test split, and
the remaining slides are further split into training and early-stopping
validation subsets.

The output split files follow the CLAM split CSV format, with three columns:
train, val, and test. These files can be passed directly to the CLAM training
pipeline to ensure that CLAM is evaluated on aligned development folds.

Usage
-----
Example:
python 2_create_splits_for_CV.py \
    --labels_csv clam_lvi_labels.csv \
    --pt_root /mnt/d/CLAM_data/LVI/pt_files/ \
    --out_dir ./splits \
    --cv_folds 5 \
    --cv_val_frac 0.15 \
    --seed 99 \
    --require_pt

Outputs
-------
- splits_0.csv ... splits_4.csv
- split_summary.csv

Notes
-----
If --require_pt is used, slides without matching .pt feature files are excluded
before the CV splits are generated.
"""


import os
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def pad_columns(train_ids, val_ids, test_ids):
    """Pad three lists to equal length for CLAM split csv format."""
    max_len = max(len(train_ids), len(val_ids), len(test_ids))
    train_col = train_ids + [None] * (max_len - len(train_ids))
    val_col = val_ids + [None] * (max_len - len(val_ids))
    test_col = test_ids + [None] * (max_len - len(test_ids))
    return pd.DataFrame({
        "train": train_col,
        "val": val_col,
        "test": test_col,
    })


def main(args):
    labels_csv = Path(args.labels_csv)
    pt_root = Path(args.pt_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_csv)

    # Basic checks
    required_cols = {"slide_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"labels csv missing required columns: {missing}")

    # Keep only slides that actually have .pt feature files
    if args.require_pt:
        df["pt_exists"] = df["slide_id"].apply(lambda x: (pt_root / f"{x}.pt").exists())
        missing_df = df.loc[~df["pt_exists"], ["slide_id"]].copy()
        n_missing = len(missing_df)

        if n_missing > 0:
            print(f"[WARN] {n_missing} slides in csv do not have matching .pt files. They will be dropped.")
            print("Missing slide_id:")
            for sid in missing_df["slide_id"].tolist():
                print("  ", sid)
        df = df[df["pt_exists"]].copy()
        df = df.drop(columns=["pt_exists"])

    # Map labels to ints if they are strings
    if df["label"].dtype == object:
        label_map = {"negative": 0, "positive": 1}
        if not set(df["label"].dropna().unique()).issubset(set(label_map.keys())):
            raise ValueError(
                f"Found unexpected label values: {sorted(df['label'].dropna().unique())}. "
                "Expected negative/positive."
            )
        df["y"] = df["label"].map(label_map).astype(int)
    else:
        df["y"] = df["label"].astype(int)

    # Drop duplicate slide_ids if any
    if df["slide_id"].duplicated().any():
        dupes = df.loc[df["slide_id"].duplicated(), "slide_id"].tolist()
        raise ValueError(f"Duplicate slide_id found in labels csv, e.g. {dupes[:10]}")

    slide_ids = df["slide_id"].tolist()
    y_all = df["y"].to_numpy(dtype=np.int64)

    skf = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.seed,
    )

    summary_rows = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(slide_ids)), y_all)):
        trainval_df = df.iloc[trainval_idx].copy()
        test_df = df.iloc[test_idx].copy()

        y_trainval = trainval_df["y"].to_numpy(dtype=np.int64)

        # Inner split:
        # from the training portion of each outer fold, hold out a validation subset
        # for CLAM model selection / early stopping.
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=args.cv_val_frac,
            random_state=args.seed + (fold_idx + 1),
            stratify=y_trainval,
        )

        train_ids = train_df["slide_id"].tolist()
        val_ids = val_df["slide_id"].tolist()
        test_ids = test_df["slide_id"].tolist()

        split_df = pad_columns(train_ids, val_ids, test_ids)
        split_path = out_dir / f"splits_{fold_idx}.csv"
        split_df.to_csv(split_path, index=False)

        summary_rows.append({
            "fold": fold_idx,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
            "pos_train": int(train_df["y"].sum()),
            "pos_val": int(val_df["y"].sum()),
            "pos_test": int(test_df["y"].sum()),
        })

        print(f"[Fold {fold_idx}] "
              f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} | "
              f"pos(train)={int(train_df['y'].sum())} "
              f"pos(val)={int(val_df['y'].sum())} "
              f"pos(test)={int(test_df['y'].sum())}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "split_summary.csv", index=False)
    print(f"\nSaved {args.cv_folds} split files to: {out_dir}")
    print(f"Saved summary to: {out_dir / 'split_summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export CLAM split csvs using the same 5-fold logic as the user's GigaPath CV."
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        required=True,
        help="Path to clam_lvi_labels.csv",
    )
    parser.add_argument(
        "--pt_root",
        type=str,
        required=True,
        help="Directory containing per-slide .pt feature files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save splits_0.csv ... splits_{k-1}.csv",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of outer CV folds",
    )
    parser.add_argument(
        "--cv_val_frac",
        type=float,
        default=0.15,
        help="Validation fraction taken from trainval in each outer fold",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=99,
        help="Random seed; matches your current CV style",
    )
    parser.add_argument(
        "--require_pt",
        action="store_true",
        help="Drop slides that do not have matching .pt files",
    )

    args = parser.parse_args()
    main(args)