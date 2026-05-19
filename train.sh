#!/bin/bash
set -e

echo "========================================================="
echo "Starting EfficientTAM Full Training & Evaluation Pipeline"
echo "Configuration: ti_512x512 (Lowest VRAM Usage)"
echo "Note: Data roots and output paths are loaded from .env"
echo "========================================================="

# 1. Check and load .env file
if [ -f .env ]; then
    # Export variables from .env to make them accessible to bash
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    echo "Please copy .env.example to .env and configure your paths."
    exit 1
fi

# Resolve paths for the evaluation script based on .env logic
# (It falls back to the shared DATA_ROOT / OUTPUT_DIR if specific ones are empty)
VAL_ROOT_VIDEO=${DATA_ROOT_VIDEO:-$DATA_ROOT}
OUT_DIR_VIDEO=${OUTPUT_DIR_VIDEO:-$OUTPUT_DIR}

if [ -z "$VAL_ROOT_VIDEO" ] || [ -z "$OUT_DIR_VIDEO" ]; then
    echo "Error: Missing video data root or output directory configuration in .env."
    echo "Please ensure DATA_ROOT_VIDEO (or DATA_ROOT) and OUTPUT_DIR_VIDEO (or OUTPUT_DIR) are set."
    exit 1
fi

# ---------------------------------------------------------
# Stage 1: Image Pretraining
# ---------------------------------------------------------
echo -e "\n[1/3] Running Stage 1: Image Pretraining..."
# --data-root, --output-dir, etc. are implicitly loaded from .env by the python script
python3 -m training.train_image \
    --config training/train_image_test

# ---------------------------------------------------------
# Stage 2: Video Fine-tuning
# ---------------------------------------------------------
echo -e "\n[2/3] Running Stage 2: Video Fine-tuning..."
# --data-root, --output-dir, and --init-from are implicitly loaded from .env
python3 -m training.train_video \
    --config training/train_video_test

# ---------------------------------------------------------
# Stage 3: Evaluation (DAVIS-style Validation)
# ---------------------------------------------------------
echo -e "\n[3/3] Running Evaluation on Validation Set..."
# tools.validate does not use dotenv, so we must explicitly pass the arguments from bash
python3 -m tools.validate \
    --config configs/efficienttam/efficienttam_ti_512x512.yaml \
    --ckpt "$OUT_DIR_VIDEO/video_final.pt" \
    --val-root "$VAL_ROOT_VIDEO" \
    --output-json "$OUT_DIR_VIDEO/results.json"

echo -e "\n========================================================="
echo "Pipeline completed successfully!"
echo "Evaluation Results saved in: $OUT_DIR_VIDEO/results.json"
echo "========================================================="
