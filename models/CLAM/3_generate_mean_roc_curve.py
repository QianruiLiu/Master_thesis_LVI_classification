"""
Plot the mean ROC curve for the CLAM baseline across development CV folds.

The script reads CLAM output pickle files, extracts ground-truth labels and
predicted class probabilities, computes one ROC curve and AUC per fold, and then
plots the mean ROC curve with ±1 standard deviation shading.

Usage
-----
Run from the project root directory:

python models/CLAM/3_generate_mean_roc_curve.py \
  --results-dir models/CLAM/results/lvi_binary_s99 \
  --out-path models/CLAM/results/lvi_binary_s99/mean_roc_curve.png \
  --n-folds 5

Outputs
-------
CV roc curve for CLAM model
"""

import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a mean ROC curve across CLAM CV split result pickle files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results/lvi_binary_s99"),
        help="Folder containing split_0_results.pkl ... split_<n-1>_results.pkl. "
             "Default: ./results/lvi_binary_s99",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("mean_roc_curve.png"),
        help="Output path for the mean ROC curve figure. Default: mean_roc_curve.png",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds / split result files to read. Default: 5",
    )
    parser.add_argument(
        "--fpr-points",
        type=int,
        default=100,
        help="Number of points in the shared FPR grid for interpolation. Default: 100",
    )
    return parser.parse_args()


def load_clam_split_pkl(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    y_true = []
    y_prob = []

    for slide_id, rec in data.items():
        label = int(rec["label"])
        prob = np.array(rec["prob"]).squeeze()  # [[a, b]] -> [a, b]
        p_pos = float(prob[1])

        y_true.append(label)
        y_prob.append(p_pos)

    return np.array(y_true), np.array(y_prob)


def plot_mean_roc_from_clam_pkls(
    results_dir: Path,
    n_folds: int = 5,
    out_path: Path = Path("mean_roc_curve.png"),
    fpr_points: int = 100,
):
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")
    if not results_dir.is_dir():
        raise NotADirectoryError(f"results_dir is not a directory: {results_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    mean_fpr = np.linspace(0, 1, fpr_points)
    tprs = []
    aucs = []

    for i in range(n_folds):
        pkl_path = results_dir / f"split_{i}_results.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Missing CLAM result file for fold {i}: {pkl_path}")

        y_true, y_prob = load_clam_split_pkl(pkl_path)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fold_auc = roc_auc_score(y_true, y_prob)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0

        tprs.append(interp_tpr)
        aucs.append(fold_auc)

        print(f"Fold {i}: AUC = {fold_auc:.4f}, n = {len(y_true)}")

    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    lower = np.clip(mean_tpr - std_tpr, 0.0, 1.0)
    upper = np.clip(mean_tpr + std_tpr, 0.0, 1.0)

    plt.figure(figsize=(6, 6), dpi=150)
    plt.plot(
        mean_fpr,
        mean_tpr,
        linewidth=2.0,
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
    )
    plt.fill_between(mean_fpr, lower, upper, alpha=0.2, label="±1 std")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, alpha=0.7)

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Mean ROC across CV folds")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"\nSaved figure to: {out_path}")
    print(f"Mean AUC = {mean_auc:.6f}")
    print(f"Std AUC  = {std_auc:.6f}")


def main():
    args = parse_args()
    plot_mean_roc_from_clam_pkls(
        results_dir=args.results_dir,
        n_folds=args.n_folds,
        out_path=args.out_path,
        fpr_points=args.fpr_points,
    )


if __name__ == "__main__":
    main()
