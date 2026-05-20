#!/usr/bin/env bash
set -eu

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT_DIR"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

TEST_ROOT="${TEST_ROOT:-${DATA_ROOT:-/tmp/efficienttam_test_data}}"
IMAGE_ROOT="${DATA_ROOT_IMAGE:-$TEST_ROOT/image}"
VIDEO_ROOT="${DATA_ROOT_VIDEO:-$TEST_ROOT/video}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_DIR:-$TEST_ROOT/runs}}"
CHECKPOINT_ROOT="$RUN_ROOT/checkpoints"
IMAGE_OUT="${OUTPUT_DIR_IMAGE:-$CHECKPOINT_ROOT/image}"
VIDEO_OUT="${OUTPUT_DIR_VIDEO:-$CHECKPOINT_ROOT/video}"
export TEST_ROOT RUN_ROOT

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-1}"
SMOKE_SIZE="${SMOKE_SIZE:-160}"
SMOKE_VALIDATE="${SMOKE_VALIDATE:-1}"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES
export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "[test_train] root=$TEST_ROOT"
echo "[test_train] image_data=$IMAGE_ROOT"
echo "[test_train] video_data=$VIDEO_ROOT"
echo "[test_train] image_out=$IMAGE_OUT"
echo "[test_train] video_out=$VIDEO_OUT"
echo "[test_train] nproc=$NPROC_PER_NODE cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[test_train] max_steps=$SMOKE_MAX_STEPS synthetic_size=$SMOKE_SIZE"
echo "[test_train] python=$PYTHON"
echo "[test_train] validate=$SMOKE_VALIDATE"

run_module() {
  module_name="$1"
  shift
  if [ "$NPROC_PER_NODE" = "1" ]; then
    $PYTHON -m "$module_name" "$@"
    return
  fi
  if ! command -v torchrun >/dev/null 2>&1; then
    echo "[test_train] ERROR: torchrun not found. Install PyTorch or use NPROC_PER_NODE=1." >&2
    exit 127
  fi
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" --module "$module_name" "$@"
}

rm -rf "$IMAGE_ROOT" "$VIDEO_ROOT" "$IMAGE_OUT" "$VIDEO_OUT"
$PYTHON data/prepare_mini_dataset.py \
  --out-root "$TEST_ROOT" \
  --image-count 4 \
  --video-count 2 \
  --frame-count 3 \
  --size "$SMOKE_SIZE"

echo "[test_train] stage 1 image smoke training"
run_module training.train_image \
  --config configs/training/_image_test \
  --data-root "$IMAGE_ROOT" \
  --output-dir "$IMAGE_OUT" \
  --max-steps "$SMOKE_MAX_STEPS" \
  --precision fp32 \
  train.batch_size=1 \
  train.accumulation_steps=2 \
  train.objects_per_image=1 \
  train.image_size=512

echo "[test_train] stage 2 video smoke training"
run_module training.train_video \
  --config configs/training/_video_test \
  --data-root "$VIDEO_ROOT" \
  --image-data-root "$IMAGE_ROOT" \
  --output-dir "$VIDEO_OUT" \
  --init-from "$IMAGE_OUT/image_final.pt" \
  --max-steps "$SMOKE_MAX_STEPS" \
  --precision fp32 \
  train.batch_size=1 \
  train.accumulation_steps=2 \
  train.clip_len=2 \
  train.objects_per_clip=1 \
  train.image_mix_prob=0.5 \
  train.image_size=512

if [ "$SMOKE_VALIDATE" = "1" ]; then
  echo "[test_train] stage 3 metrics evaluation smoke test"
  $PYTHON -m tools.validate \
    --config configs/efficienttam/efficienttam_ti_512x512.yaml \
    --ckpt "$VIDEO_OUT/video_final.pt" \
    --val-root "$VIDEO_ROOT"
else
  echo "[test_train] skipping validation because SMOKE_VALIDATE=$SMOKE_VALIDATE"
fi

echo "[test_train] done"
echo "[test_train] image checkpoint: $IMAGE_OUT/image_final.pt"
echo "[test_train] video checkpoint: $VIDEO_OUT/video_final.pt"
