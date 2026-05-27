#!/usr/bin/env bash
set -euo pipefail

# Activate env if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate gigapath

SLIDES_DIR="/mnt/d/BMM_LVI"
H5_ROOT="/mnt/d/tile_encoder_h5files"
DEVELOP_TSV="/mnt/d/labels_train.tsv"
EXTERNAL_TSV="/mnt/d/labels_external.tsv"

BASE_OUT="/mnt/d/runs/final_external_eval"


OUT_DIR="${BASE_OUT}/lr3e-3_wd1e-2_k512_seed77"

python 5_final_independent_test.py \
  --slides_dir "${SLIDES_DIR}" \
  --h5_root "${H5_ROOT}" \
  --develop_tsv "${DEVELOP_TSV}" \
  --external_tsv "${EXTERNAL_TSV}" \
  --out_dir "${OUT_DIR}" \
  --val_frac 0.2 \
  --k_max 512 \
  --tile_size 1024 \
  --margin_px 1024 \
  --roi_inside_cap 256 \
  --epochs 50 \
  --patience 10 \
  --grad_accum 4 \
  --lr_head 3e-3 \
  --weight_decay 1e-2 \
  --use_pos_weight \
  --make_heatmaps \
  --heatmap_split test \
  --heatmap_block_px 4096 \
  --heatmap_thumb_max_px 2048 \
  --roi_png_dir "/mnt/d/cutting_figures" \
  --seed 77
