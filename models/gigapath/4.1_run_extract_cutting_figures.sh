#!/usr/bin/env bash
set -euo pipefail

SLIDE_DIR="/mnt/d/BMM_LVI"
OUT_DIR="/mnt/d/cutting_figures"

mkdir -p "$OUT_DIR"

for slide_path in "$SLIDE_DIR"/*.ndpi; do
    slide_id="$(basename "$slide_path")"
    echo "[RUN] $slide_id"

    python 4_extract_cutting_figures.py \
        --slide_path "$slide_path" \
        --slide_id "$slide_id" \
        --out_dir "$OUT_DIR" \
        --level 1 \
        --margin 0
done