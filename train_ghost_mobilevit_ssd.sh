#!/bin/bash
# =============================================================================
# Train Ghost Conv MobileViT-S SSD on COCO
#
# This script trains the SSD object detector with a Ghost Conv MobileViT-S
# backbone on the MS-COCO dataset. The Ghost convolution replaces standard
# pointwise and spatial convolutions with a cheaper alternative that uses
# a primary conv + depthwise linear transforms (GhostNet-style).
#
# Usage:
#   bash scripts/train_ghost_mobilevit_ssd.sh [--num-gpus N] [--resume]
#
# Arguments:
#   --num-gpus N    Number of GPUs to use (default: 4)
#   --resume        Resume training from latest checkpoint
# =============================================================================

set -e

# --------------- Configuration ---------------
NUM_GPUS=${1:-2}
RESUME_FLAG=""
if [[ "$*" == *"--resume"* ]]; then
    RESUME_FLAG="--common.auto-resume true"
fi

CONFIG_FILE="config/detection/ssd_coco/mobilevit_ghost.yaml"
RESULTS_DIR="results/ghost_mobilevit_ssd"

# Validate config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    exit 1
fi

# --------------- Environment Setup ---------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "============================================"
echo " Training Ghost Conv MobileViT-S SSD on COCO"
echo "============================================"
echo " Config:      $CONFIG_FILE"
echo " Results dir: $RESULTS_DIR"
echo " Num GPUs:    $NUM_GPUS"
echo " Resume:      $([[ -n "$RESUME_FLAG" ]] && echo "yes" || echo "no")"
echo "============================================"
echo ""

# --------------- Launch Training ---------------
if [ "$NUM_GPUS" -gt 1 ]; then
    # Distributed training
    torchrun --nproc_per_node="$NUM_GPUS" \
        main_train.py \
        --common.config-file "$CONFIG_FILE" \
        --common.results-loc "$RESULTS_DIR" \
        $RESUME_FLAG
else
    # Single GPU training
    python main_train.py \
        --common.config-file "$CONFIG_FILE" \
        --common.results-loc "$RESULTS_DIR" \
        $RESUME_FLAG
fi

echo ""
echo "Training completed. Results saved to $RESULTS_DIR"
