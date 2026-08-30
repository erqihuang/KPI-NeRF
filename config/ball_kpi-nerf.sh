#!/usr/bin/env bash
# Example evaluation command for a trained ball scene.
# Update SCENE_NAME, CKPT_DIR and GT_CSV_FILE before running.
set -euo pipefail

BASE_DIR="./"
SCENE_NAME="./data/<date>/ball_scene"
CKPT_DIR="./logs/<run-directory>"
GT_CSV_FILE="./data/GT/ball.csv"
GPU_ID="${GPU_ID:-0}"

python tasks/any_folder_spec_klensPlus_RGB_coding/spiral_Tcomp.py \
    --base_dir "$BASE_DIR" \
    --scene_name "$SCENE_NAME" \
    --ckpt_dir "$CKPT_DIR" \
    --gt_csv_file "$GT_CSV_FILE" \
    --gpu_id "$GPU_ID" \
    --ls_factor 3.2 \
    --ls_file xenon_c29 \
    --resize_ratio 1 \
    --bf_num 5 \
    --spec_chnls 29 \
    --gamma 2.2 \
    --init_basis DCT \
    --act linear \
    --spiral_axis_scale 1 1 1 \
    --fx_only \
    --stare \
    --spiral
