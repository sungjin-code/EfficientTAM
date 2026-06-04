#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
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

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
VARIANT="${VARIANT:-ti}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES

if [ -n "${DATA_ROOT_IMAGE:-}" ]; then
    DATA_ROOT_IMAGE_RESOLVED="$DATA_ROOT_IMAGE"
elif [ -n "${DATA_ROOT:-}" ] && [ -d "$DATA_ROOT/sa1b" ]; then
    DATA_ROOT_IMAGE_RESOLVED="$DATA_ROOT/sa1b"
else
    DATA_ROOT_IMAGE_RESOLVED="${DATA_ROOT:-}"
fi

if [ -n "${DATA_ROOT_VIDEO:-}" ]; then
    DATA_ROOT_VIDEO_RESOLVED="$DATA_ROOT_VIDEO"
elif [ -n "${DATA_ROOT:-}" ] && [ -d "$DATA_ROOT/sav" ]; then
    DATA_ROOT_VIDEO_RESOLVED="$DATA_ROOT/sav"
else
    DATA_ROOT_VIDEO_RESOLVED="${DATA_ROOT:-}"
fi
if [ -n "${VAL_ROOT_DAVIS:-${VAL_ROOT_VIDEO:-}}" ]; then
    VAL_ROOT_DAVIS_RESOLVED="${VAL_ROOT_DAVIS:-${VAL_ROOT_VIDEO:-}}"
elif [ -n "${DATA_ROOT:-}" ] && [ -d "$DATA_ROOT/DAVIS" ]; then
    VAL_ROOT_DAVIS_RESOLVED="$DATA_ROOT/DAVIS"
else
    VAL_ROOT_DAVIS_RESOLVED=""
fi

vos_layout_root() {
    local root="$1"
    local split
    if [ -z "$root" ] || [ ! -d "$root" ]; then
        return
    fi
    if { [ -d "$root/JPEGImages" ] && [ -d "$root/Annotations" ]; } \
        || { [ -d "$root/JPEGImages/480p" ] && [ -d "$root/Annotations/480p" ]; } \
        || { [ -d "$root/JPEGImages_24fps" ] && [ -d "$root/Annotations_6fps" ]; }; then
        echo "$root"
        return
    fi
    for split in valid val test train sav_test; do
        if { [ -d "$root/$split/JPEGImages" ] && [ -d "$root/$split/Annotations" ]; } \
            || { [ -d "$root/$split/JPEGImages_24fps" ] && [ -d "$root/$split/Annotations_6fps" ]; }; then
            echo "$root/$split"
            return
        fi
    done
    echo "$root"
}

resolve_optional_val_root() {
    local explicit="$1"
    local fallback="$2"
    if [ -n "$explicit" ]; then
        vos_layout_root "$explicit"
    elif [ -n "${DATA_ROOT:-}" ] && [ -d "$DATA_ROOT/$fallback" ]; then
        vos_layout_root "$DATA_ROOT/$fallback"
    fi
}

VAL_ROOT_MOSE_RESOLVED="$(resolve_optional_val_root "${VAL_ROOT_MOSE:-}" "MOSE")"
VAL_ROOT_LVOS_RESOLVED="$(resolve_optional_val_root "${VAL_ROOT_LVOS:-}" "LVOS")"
VAL_ROOT_SAV_RESOLVED="$(resolve_optional_val_root "${VAL_ROOT_SAV:-}" "SA-V-test")"
VAL_ROOT_YTVOS_RESOLVED="$(resolve_optional_val_root "${VAL_ROOT_YTVOS:-}" "YTVOS2019")"
RUN_ROOT="${OUTPUT_DIR:-runs}"
OUTPUT_DIR_IMAGE_RESOLVED="${OUTPUT_DIR_IMAGE:-$RUN_ROOT/image_${VARIANT}}"
OUTPUT_DIR_VIDEO_RESOLVED="${OUTPUT_DIR_VIDEO:-$RUN_ROOT/video_${VARIANT}}"
EVAL_DIR="$OUTPUT_DIR_VIDEO_RESOLVED/eval"
IMAGE_ARTIFACT="$OUTPUT_DIR_IMAGE_RESOLVED/image_training_artifact.json"
VIDEO_ARTIFACT="$OUTPUT_DIR_VIDEO_RESOLVED/video_training_artifact.json"

case "$VARIANT" in
    ti)
        IMAGE_CONFIG="training/train_image_ti"
        VIDEO_CONFIG="training/train_video_ti"
        VALIDATE_CONFIG="configs/efficienttam/efficienttam_ti.yaml"
        IMAGE_BATCH_PER_GPU=16
        VIDEO_BATCH_PER_GPU=4
        ;;
    s)
        IMAGE_CONFIG="training/train_image_s"
        VIDEO_CONFIG="training/train_video_s"
        VALIDATE_CONFIG="configs/efficienttam/efficienttam_s.yaml"
        IMAGE_BATCH_PER_GPU=8
        VIDEO_BATCH_PER_GPU=2
        ;;
    *)
        echo "Error: unsupported VARIANT='$VARIANT'. Use VARIANT=ti or VARIANT=s."
        exit 1
        ;;
esac

calc_accumulation_steps() {
    local per_gpu_batch="$1"
    local micro_global=$((per_gpu_batch * NPROC_PER_NODE))
    if [ $((TARGET_GLOBAL_BATCH % micro_global)) -ne 0 ]; then
        echo "Error: TARGET_GLOBAL_BATCH=$TARGET_GLOBAL_BATCH is not divisible by micro global batch=$micro_global." >&2
        exit 1
    fi
    echo $((TARGET_GLOBAL_BATCH / micro_global))
}

IMAGE_ACCUMULATION_STEPS="$(calc_accumulation_steps "$IMAGE_BATCH_PER_GPU")"
VIDEO_ACCUMULATION_STEPS="$(calc_accumulation_steps "$VIDEO_BATCH_PER_GPU")"

if [ -z "$DATA_ROOT_IMAGE_RESOLVED" ] || [ -z "$DATA_ROOT_VIDEO_RESOLVED" ]; then
    echo "Error: Missing DATA_ROOT_IMAGE/DATA_ROOT_VIDEO, or shared DATA_ROOT, in .env."
    exit 1
fi

if [ ! -d "$DATA_ROOT_IMAGE_RESOLVED/images" ] || [ ! -d "$DATA_ROOT_IMAGE_RESOLVED/masks" ]; then
    echo "Error: Image data directory '$DATA_ROOT_IMAGE_RESOLVED' is missing images/ or masks/."
    exit 1
fi

if [ ! -d "$DATA_ROOT_VIDEO_RESOLVED/JPEGImages" ] || [ ! -d "$DATA_ROOT_VIDEO_RESOLVED/Annotations" ]; then
    echo "Error: Video data directory '$DATA_ROOT_VIDEO_RESOLVED' is missing JPEGImages/ or Annotations/."
    exit 1
fi

if [ -z "$VAL_ROOT_DAVIS_RESOLVED" ] \
    && [ -z "$VAL_ROOT_MOSE_RESOLVED" ] \
    && [ -z "$VAL_ROOT_LVOS_RESOLVED" ] \
    && [ -z "$VAL_ROOT_SAV_RESOLVED" ] \
    && [ -z "$VAL_ROOT_YTVOS_RESOLVED" ]; then
    echo "Error: set at least one VOS validation root:"
    echo "VAL_ROOT_DAVIS, VAL_ROOT_MOSE, VAL_ROOT_LVOS, VAL_ROOT_SAV, or VAL_ROOT_YTVOS."
    exit 1
fi

check_vos_root() {
    local name="$1"
    local root="$2"
    if [ -z "$root" ]; then
        return
    fi
    if { [ -d "$root/JPEGImages" ] && [ -d "$root/Annotations" ]; } \
        || { [ -d "$root/JPEGImages/480p" ] && [ -d "$root/Annotations/480p" ]; } \
        || { [ -d "$root/JPEGImages_24fps" ] && [ -d "$root/Annotations_6fps" ]; } \
        || { [ -d "$root/valid/JPEGImages" ] && [ -d "$root/valid/Annotations" ]; } \
        || { [ -d "$root/val/JPEGImages" ] && [ -d "$root/val/Annotations" ]; } \
        || { [ -d "$root/test/JPEGImages" ] && [ -d "$root/test/Annotations" ]; } \
        || { [ -d "$root/valid/JPEGImages_24fps" ] && [ -d "$root/valid/Annotations_6fps" ]; } \
        || { [ -d "$root/val/JPEGImages_24fps" ] && [ -d "$root/val/Annotations_6fps" ]; } \
        || { [ -d "$root/test/JPEGImages_24fps" ] && [ -d "$root/test/Annotations_6fps" ]; }; then
        return
    fi
    echo "Error: $name='$root' is not a supported VOS validation layout."
    echo "Expected JPEGImages/Annotations, DAVIS JPEGImages/480p + Annotations/480p,"
    echo "SA-V JPEGImages_24fps + Annotations_6fps, or those layouts under valid/val/test."
    exit 1
}

check_vos_root "VAL_ROOT_DAVIS" "$VAL_ROOT_DAVIS_RESOLVED"
check_vos_root "VAL_ROOT_MOSE" "$VAL_ROOT_MOSE_RESOLVED"
check_vos_root "VAL_ROOT_LVOS" "$VAL_ROOT_LVOS_RESOLVED"
check_vos_root "VAL_ROOT_SAV" "$VAL_ROOT_SAV_RESOLVED"
check_vos_root "VAL_ROOT_YTVOS" "$VAL_ROOT_YTVOS_RESOLVED"

if [ -n "${SA23_ROOT:-}" ] && { [ ! -d "$SA23_ROOT/images" ] || [ ! -d "$SA23_ROOT/masks" ]; }; then
    echo "Error: SA23_ROOT='$SA23_ROOT' is missing images/ or masks/."
    exit 1
fi

echo "========================================================="
echo "Starting EfficientTAM paper training pipeline"
echo "Variant: $VARIANT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "Python: $PYTHON"
echo "Target global batch: $TARGET_GLOBAL_BATCH"
echo "Stage 1: per-GPU batch=$IMAGE_BATCH_PER_GPU accumulation=$IMAGE_ACCUMULATION_STEPS"
echo "Stage 2: per-GPU batch=$VIDEO_BATCH_PER_GPU accumulation=$VIDEO_ACCUMULATION_STEPS"
echo "Image output: $OUTPUT_DIR_IMAGE_RESOLVED"
echo "Video output: $OUTPUT_DIR_VIDEO_RESOLVED"
echo "Evaluation output: $EVAL_DIR"
echo "Image artifact: $IMAGE_ARTIFACT"
echo "Video artifact: $VIDEO_ARTIFACT"
echo "========================================================="

run_module() {
    local module_name="$1"
    shift
    if [ "$NPROC_PER_NODE" = "1" ]; then
        "$PYTHON" -m "$module_name" "$@"
        return
    fi
    if command -v torchrun >/dev/null 2>&1; then
        torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" --module "$module_name" "$@"
        return
    fi
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" --module "$module_name" "$@"
}

run_module training.train_image \
    --config "$IMAGE_CONFIG" \
    --data-root "$DATA_ROOT_IMAGE_RESOLVED" \
    --output-dir "$OUTPUT_DIR_IMAGE_RESOLVED" \
    "train.accumulation_steps=$IMAGE_ACCUMULATION_STEPS"

if [ ! -f "$IMAGE_ARTIFACT" ] || [ ! -f "$OUTPUT_DIR_IMAGE_RESOLVED/image_final.pt" ]; then
    echo "Error: Stage 1 did not produce expected artifact/checkpoint." >&2
    echo "Expected: $IMAGE_ARTIFACT" >&2
    echo "Expected: $OUTPUT_DIR_IMAGE_RESOLVED/image_final.pt" >&2
    exit 1
fi

run_module training.train_video \
    --config "$VIDEO_CONFIG" \
    --data-root "$DATA_ROOT_VIDEO_RESOLVED" \
    --image-data-root "$DATA_ROOT_IMAGE_RESOLVED" \
    --output-dir "$OUTPUT_DIR_VIDEO_RESOLVED" \
    --init-from "$OUTPUT_DIR_IMAGE_RESOLVED/image_final.pt" \
    "train.accumulation_steps=$VIDEO_ACCUMULATION_STEPS"

if [ ! -f "$VIDEO_ARTIFACT" ] || [ ! -f "$OUTPUT_DIR_VIDEO_RESOLVED/video_final.pt" ]; then
    echo "Error: Stage 2 did not produce expected artifact/checkpoint." >&2
    echo "Expected: $VIDEO_ARTIFACT" >&2
    echo "Expected: $OUTPUT_DIR_VIDEO_RESOLVED/video_final.pt" >&2
    exit 1
fi

mkdir -p "$EVAL_DIR"

if [ -n "$VAL_ROOT_DAVIS_RESOLVED" ]; then
    export VAL_ROOT_DAVIS="$VAL_ROOT_DAVIS_RESOLVED"
fi
if [ -n "$VAL_ROOT_MOSE_RESOLVED" ]; then
    export VAL_ROOT_MOSE="$VAL_ROOT_MOSE_RESOLVED"
fi
if [ -n "$VAL_ROOT_LVOS_RESOLVED" ]; then
    export VAL_ROOT_LVOS="$VAL_ROOT_LVOS_RESOLVED"
fi
if [ -n "$VAL_ROOT_SAV_RESOLVED" ]; then
    export VAL_ROOT_SAV="$VAL_ROOT_SAV_RESOLVED"
fi
if [ -n "$VAL_ROOT_YTVOS_RESOLVED" ]; then
    export VAL_ROOT_YTVOS="$VAL_ROOT_YTVOS_RESOLVED"
fi

$PYTHON -m tools.validate_vos_suite \
    --config "$VALIDATE_CONFIG" \
    --ckpt "$OUTPUT_DIR_VIDEO_RESOLVED/video_final.pt" \
    --output-json "$EVAL_DIR/vos_suite.json"

if [ -n "${SA23_ROOT:-}" ]; then
    $PYTHON -m tools.validate_image_miou \
        --config "$VALIDATE_CONFIG" \
        --ckpt "$OUTPUT_DIR_VIDEO_RESOLVED/video_final.pt" \
        --root "$SA23_ROOT" \
        --output-json "$EVAL_DIR/sa23_miou.json"
else
    echo "Skipping SA-23 mIoU because SA23_ROOT is not set."
fi

echo "========================================================="
echo "Pipeline completed successfully."
echo "Evaluation results: $EVAL_DIR"
echo "Image artifact: $IMAGE_ARTIFACT"
echo "Video artifact: $VIDEO_ARTIFACT"
echo "========================================================="
