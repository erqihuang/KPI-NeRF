#!/usr/bin/env bash
# bash config/ajar_kpi-nerf.sh
set -euo pipefail

BASE_DIR="./"
SCENE_NAME="./data/20250909/exr_ajar"
ALIAS="ajar_kpi-nerf"

TRAIN_GPU_ID=3
EVAL_GPU_ID="${TRAIN_GPU_ID}"

TRAIN_SCRIPT="tasks/any_folder_spec_klensPlus_RGB_coding/train_gt.py"
SPIRAL_SCRIPT="tasks/any_folder_spec_klensPlus_RGB_coding/spiral_gt.py"
LOG_FILE="logs/ajar_kpi-nerf.out"

NERF_LR=0.001
RAND_SEED=17
RESIZE_RATIO=1
NUM_SAMPLE=128
SPEC_CHNLS=29
LS_FACTOR=1.0
BF_NUM=5
INIT_BASIS="DCT"
EPOCH=10000
FOCAL_ORDER=2
GAMMA=2.2

mkdir -p logs

PYTHONUNBUFFERED=1 python -u "$TRAIN_SCRIPT" \
    --base_dir "$BASE_DIR" \
    --scene_name "$SCENE_NAME" \
    --gpu_id "$TRAIN_GPU_ID" \
    --nerf_lr "$NERF_LR" \
    --rand_seed "$RAND_SEED" \
    --resize_ratio "$RESIZE_RATIO" \
    --num_sample "$NUM_SAMPLE" \
    --spec_chnls "$SPEC_CHNLS" \
    --ls_factor "$LS_FACTOR" \
    --bf_num "$BF_NUM" \
    --train_rand_rows 32 \
    --train_rand_cols 32 \
    --init_basis "$INIT_BASIS" \
    --epoch "$EPOCH" \
    --learn_t \
    --learn_R \
    --learn_focal \
    --fx_only \
    --linear_tonemapping \
    --focal_order "$FOCAL_ORDER" \
    --alias "$ALIAS" \
    2>&1 | tee "$LOG_FILE"

# Recreate train_gt.py::gen_detail_name(args) prefix in bash, then pick latest timestamped run.
LOG_ROOT="logs/any_folder_spec/${SCENE_NAME}"
RUN_PREFIX="lr_${NERF_LR}_gpu${TRAIN_GPU_ID}_seed_${RAND_SEED}_resize_${RESIZE_RATIO}_Nsam_${NUM_SAMPLE}_specChnls_${SPEC_CHNLS}_bfChnls_${BF_NUM}_lsfactor_${LS_FACTOR}_${ALIAS}_"
CKPT_DIR=$(ls -dt "${LOG_ROOT}/${RUN_PREFIX}"* 2>/dev/null | head -n 1)

if [[ -z "${CKPT_DIR}" ]]; then
    echo "[ERROR] Cannot resolve ckpt_dir under ${LOG_ROOT} with prefix ${RUN_PREFIX}" >&2
    exit 1
fi

echo "[INFO] Resolved ckpt_dir: ${CKPT_DIR}"
echo "[INFO] Train GPU: ${TRAIN_GPU_ID}, Eval GPU: ${EVAL_GPU_ID}"

PYTHONUNBUFFERED=1 python -u "$SPIRAL_SCRIPT" \
    --base_dir "$BASE_DIR" \
    --scene_name "$SCENE_NAME" \
    --gpu_id "$EVAL_GPU_ID" \
    --resize_ratio "$RESIZE_RATIO" \
    --ls_factor "$LS_FACTOR" \
    --bf_num "$BF_NUM" \
    --spec_chnls "$SPEC_CHNLS" \
    --gamma "$GAMMA" \
    --ckpt_dir "$CKPT_DIR" \
    --init_basis "$INIT_BASIS" \
    --spiral_axis_scale 1 1 1 \
    --focal_order "$FOCAL_ORDER" \
    --fx_only \
    --stare
    # --spiral
