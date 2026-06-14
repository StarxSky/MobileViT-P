#!/bin/bash
# =============================================================================
# Evaluate Ghost Conv MobileViT-S SSD on COCO
#
# This script evaluates a trained Ghost Conv MobileViT-S SSD model on the
# MS-COCO validation set, computing standard COCO mAP metrics.
#
# Usage:
#   bash scripts/test_ghost_mobilevit_ssd.sh <checkpoint_path> [--num-gpus N]
#
# Arguments:
#   checkpoint_path  Path to the trained model checkpoint (.pt file)
#   --num-gpus N     Number of GPUs to use (default: 1)
#   --image-folder   Run inference on an image folder instead of COCO val set
#   --image-path     Run inference on a single image
# =============================================================================

set -e

# --------------- Parse Arguments ---------------
if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/test_ghost_mobilevit_ssd.sh <checkpoint_path> [--num-gpus N] [--image-folder PATH] [--image-path PATH]"
    exit 1
fi

CHECKPOINT_PATH="$1"
shift

NUM_GPUS=1
EVAL_MODE="validation_set"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --image-folder)
            EVAL_MODE="image_folder"
            EXTRA_ARGS="--evaluation.detection.path $2"
            shift 2
            ;;
        --image-path)
            EVAL_MODE="single_image"
            EXTRA_ARGS="--evaluation.detection.path $2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 1
            ;;
    esac
done

CONFIG_FILE="config/detection/ssd_coco/mobilevit_ghost.yaml"
RESULTS_DIR="results/ghost_mobilevit_ssd_eval"

# Validate inputs
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "============================================"
echo " Evaluating Ghost Conv MobileViT-S SSD"
echo "============================================"
echo " Config:       $CONFIG_FILE"
echo " Checkpoint:   $CHECKPOINT_PATH"
echo " Eval mode:    $EVAL_MODE"
echo " Results dir:  $RESULTS_DIR"
echo " Num GPUs:     $NUM_GPUS"
echo "============================================"
echo ""

# --------------- Launch Evaluation ---------------
if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$NUM_GPUS" \
        main_eval.py \
        --common.config-file "$CONFIG_FILE" \
        --common.results-loc "$RESULTS_DIR" \
        --model.detection.pretrained "$CHECKPOINT_PATH" \
        --evaluation.detection.resize-input-images \
        --evaluation.detection.mode "$EVAL_MODE" \
        $EXTRA_ARGS
else
    python main_eval.py \
        --common.config-file "$CONFIG_FILE" \
        --common.results-loc "$RESULTS_DIR" \
        --model.detection.pretrained "$CHECKPOINT_PATH" \
        --evaluation.detection.resize-input-images \
        --evaluation.detection.mode "$EVAL_MODE" \
        $EXTRA_ARGS
fi

echo ""
echo "Evaluation completed. Results saved to $RESULTS_DIR"
