#!/usr/bin/env bash
set -euo pipefail

# Activate env if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate gigapath

SLIDES_DIR="/mnt/d/BMM_LVI"
H5_ROOT="/mnt/d/tile_encoder_h5files"
LABELS_TSV="/mnt/d/labels_train.tsv"  

BASE_OUT="/mnt/d/runs/last_try_run"
SEED=99


OUT_DIR="${BASE_OUT}/inside_roi_sampling_3e-3_1e-2"
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
    --lr_head 3e-3 \
    --weight_decay 1e-2 \
    --use_pos_weight \
    --seed ${SEED} \
    --roi_inside_cap 256 \
    --make_heatmaps \
    --heatmap_split test \
    --heatmap_block_px 4096 \
    --heatmap_thumb_max_px 2048 \
    --roi_png_dir /mnt/d/cutting_figures
