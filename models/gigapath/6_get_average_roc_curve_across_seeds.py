"""
This script plots the mean independent-test ROC curve across five independently
trained model seeds. For each seed, it reads the saved final_metrics.json file,
extracts the external-test ground-truth labels and predicted probabilities, and
computes a seed-specific ROC curve and AUC.

To summarize performance across seeds, each ROC curve is interpolated onto a
shared false-positive-rate grid. The script then calculates the mean TPR and
standard deviation across seeds, and plots the mean ROC curve with ±1 standard
deviation shading.

Usage
-----
Run from the project root directory:

    python models/gigapath/6_get_average_roc_curve_across_seeds.py \
        --json-files \
            /mnt/d/runs/final_external_eval/lr3e-3_wd1e-2_k512_seed35/final_metrics.json \
            /mnt/d/runs/final_external_eval/lr3e-3_wd1e-2_k512_seed42/final_metrics.json \
            /mnt/d/runs/final_external_eval/lr3e-3_wd1e-2_k512_seed66/final_metrics.json \
            /mnt/d/runs/final_external_eval/lr3e-3_wd1e-2_k512_seed77/final_metrics.json \
            /mnt/d/runs/final_external_eval/lr3e-3_wd1e-2_k512_seed99/final_metrics.json \
        --out-file /mnt/d/runs/final_external_eval/mean_roc_across_5seeds.png

Outputs
-------
5 seeds ROC plot
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


DEFAULT_JSON_FILES = [
    "/mnt/d/runs/final_external_eval/lr3e-3_wd1e-2_k512_seed35/final_metrics.json",
    "/mnt/d/runs/final_external_eval/lr3e-3_wd1e-2_k512_seed42/final_metrics.json",
    "/mnt/d/runs/final_external_eval/lr3e-3_wd1e-2_k512_seed66/final_metrics.json",
    "/mnt/d/runs/final_external_eval/lr3e-3_wd1e-2_k512_seed77/final_metrics.json",
    "/mnt/d/runs/final_external_eval/lr3e-3_wd1e-2_k512_seed99/final_metrics.json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot mean ROC curve across multiple final_metrics.json files."
    )
    parser.add_argument(
        "--json-files",
        nargs="+",
        default=DEFAULT_JSON_FILES,
        help="List of final_metrics.json files. Default: the five original seed paths.",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("mean_roc_across_5seeds.png"),
        help="Output PNG path. Default: mean_roc_across_5seeds.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Independent testing ROC curve across 5 seeds",
        help="Figure title. Default: Independent testing ROC curve across 5 seeds",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    json_files = [Path(p) for p in args.json_files]
    for jf in json_files:
        if not jf.exists():
            raise FileNotFoundError(f"final_metrics.json does not exist: {jf}")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)

    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []

    plt.figure(figsize=(6, 6))

    for i, jf in enumerate(json_files):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        y_true = np.array(data["test_y_true"])
        y_prob = np.array(data["test_y_prob"])

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        # Optional: draw original ROC for each seed.
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"Seed {i + 1} AUC={roc_auc:.3f}")

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.plot(
        mean_fpr,
        mean_tpr,
        lw=2,
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
    )

    plt.fill_between(
        mean_fpr,
        tpr_lower,
        tpr_upper,
        alpha=0.2,
        label="±1 std",
    )

    plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(args.title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.out_file, dpi=300)

    print("Saved ROC plot to:", args.out_file)
    print("Per-seed AUCs:", [round(x, 4) for x in aucs])
    print(f"Mean AUC = {np.mean(aucs):.4f}")
    print(f"Std AUC  = {np.std(aucs):.4f}")


if __name__ == "__main__":
    main()
