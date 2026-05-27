#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
else
    echo "Error: .env file not found."
    echo "Please copy .env.example to .env and configure your paths."
    exit 1
fi

PYTHON="${PYTHON:-python3}"
DATA_ROOT_RESOLVED="${DATA_ROOT:-$ROOT_DIR/datasets}"
OUTPUT_ROOT_RESOLVED="${OUTPUT_DIR:-$ROOT_DIR/runs}"
SMOKE_ROOT="${SMOKE_ROOT:-$OUTPUT_ROOT_RESOLVED/smoke_test}"

RUN_DOWNLOAD="${RUN_DOWNLOAD:-1}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"

echo "========================================================="
echo "EfficientTAM full pipeline"
echo "DATA_ROOT=$DATA_ROOT_RESOLVED"
echo "OUTPUT_DIR=$OUTPUT_ROOT_RESOLVED"
echo "SMOKE_ROOT=$SMOKE_ROOT"
echo "RUN_DOWNLOAD=$RUN_DOWNLOAD"
echo "RUN_SMOKE_TEST=$RUN_SMOKE_TEST"
echo "RUN_TRAIN=$RUN_TRAIN"
echo "PYTHON=$PYTHON"
echo "========================================================="

require_file() {
    local path="$1"
    local label="$2"
    if [ ! -s "$path" ]; then
        echo "Error: missing $label: $path" >&2
        exit 1
    fi
}

if [ "$RUN_DOWNLOAD" = "1" ]; then
    echo "[main] Step 1/3: data download and preparation"
    ./download.sh
else
    echo "[main] Step 1/3: skipping data download because RUN_DOWNLOAD=$RUN_DOWNLOAD"
fi

if [ "$RUN_SMOKE_TEST" = "1" ]; then
    echo "[main] Step 2/3: smoke test"
    TEST_ROOT="$SMOKE_ROOT/data" \
    RUN_ROOT="$SMOKE_ROOT/runs" \
    SMOKE_NPROC_PER_NODE="${SMOKE_NPROC_PER_NODE:-1}" \
    SMOKE_CUDA_VISIBLE_DEVICES="${SMOKE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}" \
    NPROC_PER_NODE="${SMOKE_NPROC_PER_NODE:-1}" \
    CUDA_VISIBLE_DEVICES="${SMOKE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}" \
    SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-1}" \
    SMOKE_VALIDATE="${SMOKE_VALIDATE:-1}" \
    PYTHON="$PYTHON" \
        ./smoke_test.sh

    require_file "$SMOKE_ROOT/runs/checkpoints/image/image_training_artifact.json" \
        "smoke image training artifact"
    require_file "$SMOKE_ROOT/runs/checkpoints/image/image_final.pt" \
        "smoke image final checkpoint"
    require_file "$SMOKE_ROOT/runs/checkpoints/video/video_training_artifact.json" \
        "smoke video training artifact"
    require_file "$SMOKE_ROOT/runs/checkpoints/video/video_final.pt" \
        "smoke video final checkpoint"
else
    echo "[main] Step 2/3: skipping smoke test because RUN_SMOKE_TEST=$RUN_SMOKE_TEST"
fi

if [ "$RUN_TRAIN" = "1" ]; then
    echo "[main] Step 3/3: paper training"
    ./train.sh

    VARIANT_RESOLVED="${VARIANT:-ti}"
    IMAGE_OUT="${OUTPUT_DIR_IMAGE:-$OUTPUT_ROOT_RESOLVED/image_${VARIANT_RESOLVED}}"
    VIDEO_OUT="${OUTPUT_DIR_VIDEO:-$OUTPUT_ROOT_RESOLVED/video_${VARIANT_RESOLVED}}"
    require_file "$IMAGE_OUT/image_training_artifact.json" "image training artifact"
    require_file "$IMAGE_OUT/image_final.pt" "image final checkpoint"
    require_file "$VIDEO_OUT/video_training_artifact.json" "video training artifact"
    require_file "$VIDEO_OUT/video_final.pt" "video final checkpoint"
else
    echo "[main] Step 3/3: skipping paper training because RUN_TRAIN=$RUN_TRAIN"
fi

echo "========================================================="
echo "EfficientTAM full pipeline completed"
echo "========================================================="
