#!/usr/bin/env bash
set -euo pipefail

# Activate env if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate gigapath

SLIDES_DIR="/mnt/d/BMM_LVI"
H5_ROOT="/mnt/d/tile_encoder_h5files"
LABELS_TSV="/mnt/d/labels_train.tsv"

BASE_OUT="/mnt/d/runs/final_tuning"
SEED=99

WD_LIST=(1 5e-1 1e-1 3e-3 1e-3 1e-2)
LR_LIST=(3e-3 1e-2)
for WD in "${WD_LIST[@]}"; do
    for LR in "${LR_LIST[@]}"; do

      OUT_DIR="${BASE_OUT}/insideROI_k512_lr${LR}_wd${WD}_seed${SEED}"

      echo "============================================================"
      echo "Running: k_max=512, lr_head=${LR}, weight_decay=${WD}, seed=${SEED}"
      echo "Output: ${OUT_DIR}"
      echo "============================================================"

      python 3_development_model_training_5foldCV.py \
        --slides_dir "${SLIDES_DIR}" \
        --h5_root "${H5_ROOT}" \
        --labels_tsv "${LABELS_TSV}" \
        --out_dir "${OUT_DIR}" \
        --cv_folds 5 \
        --cv_val_frac 0.15 \
        --k_max 512 \
        --tile_size 1024 \
        --margin_px 1024 \
        --epochs 50 \
        --patience 10 \
        --grad_accum 4 \
        --lr_head "${LR}" \
        --weight_decay "${WD}" \
        --use_pos_weight \
        --seed "${SEED}" \
        --roi_inside_cap 256

    done
  done