#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

PYTHON="${PYTHON:-python3}"
VARIANT="${SMOKE_VARIANT:-${VARIANT:-ti}}"
NPROC_PER_NODE="${SMOKE_NPROC_PER_NODE:-1}"
CUDA_VISIBLE_DEVICES="${SMOKE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-1}"
SMOKE_IMAGE_SIZE="${SMOKE_IMAGE_SIZE:-512}"
SMOKE_CLIP_LEN="${SMOKE_CLIP_LEN:-2}"
SMOKE_VALIDATE="${SMOKE_VALIDATE:-1}"
SMOKE_IMAGE_EVAL="${SMOKE_IMAGE_EVAL:-1}"
SMOKE_USE_PAPER_CONFIGS="${SMOKE_USE_PAPER_CONFIGS:-0}"
SMOKE_MAX_VAL_VIDEOS="${SMOKE_MAX_VAL_VIDEOS:-1}"
SMOKE_MAX_VAL_IMAGES="${SMOKE_MAX_VAL_IMAGES:-1}"
SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-${OUTPUT_DIR:-$ROOT_DIR/runs}/smoke}"
IMAGE_OUT="${SMOKE_OUTPUT_DIR_IMAGE:-$SMOKE_OUTPUT_ROOT/image_${VARIANT}}"
VIDEO_OUT="${SMOKE_OUTPUT_DIR_VIDEO:-$SMOKE_OUTPUT_ROOT/video_${VARIANT}}"
EVAL_DIR="${SMOKE_EVAL_DIR:-$VIDEO_OUT/eval}"
IMAGE_ARTIFACT="$IMAGE_OUT/image_training_artifact.json"
VIDEO_ARTIFACT="$VIDEO_OUT/video_training_artifact.json"

export CUDA_VISIBLE_DEVICES
export WANDB_MODE="${WANDB_MODE:-disabled}"

if [ -n "${DATA_ROOT_IMAGE:-}" ]; then
    IMAGE_ROOT="$DATA_ROOT_IMAGE"
elif [ -n "${DATA_ROOT:-}" ] && [ -d "$DATA_ROOT/sa1b" ]; then
    IMAGE_ROOT="$DATA_ROOT/sa1b"
else
    IMAGE_ROOT="${DATA_ROOT:-}"
fi

if [ -n "${DATA_ROOT_VIDEO:-}" ]; then
    VIDEO_ROOT="$DATA_ROOT_VIDEO"
elif [ -n "${DATA_ROOT:-}" ] && [ -d "$DATA_ROOT/sav" ]; then
    VIDEO_ROOT="$DATA_ROOT/sav"
elif [ -n "${DATA_ROOT:-}" ] && [ -d "$DATA_ROOT/sav_sample" ]; then
    VIDEO_ROOT="$DATA_ROOT/sav_sample"
else
    VIDEO_ROOT="${DATA_ROOT:-}"
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

case "$VARIANT" in
    ti)
        PAPER_IMAGE_CONFIG="training/train_image_ti"
        PAPER_VIDEO_CONFIG="training/train_video_ti"
        VALIDATE_CONFIG="configs/efficienttam/efficienttam_ti_512x512.yaml"
        ;;
    s)
        PAPER_IMAGE_CONFIG="training/train_image_s"
        PAPER_VIDEO_CONFIG="training/train_video_s"
        VALIDATE_CONFIG="configs/efficienttam/efficienttam_s_512x512.yaml"
        ;;
    *)
        echo "[smoke_test] ERROR: unsupported VARIANT='$VARIANT'. Use VARIANT=ti or VARIANT=s." >&2
        exit 1
        ;;
esac

if [ "$SMOKE_USE_PAPER_CONFIGS" = "1" ]; then
    IMAGE_CONFIG="${SMOKE_IMAGE_CONFIG:-$PAPER_IMAGE_CONFIG}"
    VIDEO_CONFIG="${SMOKE_VIDEO_CONFIG:-$PAPER_VIDEO_CONFIG}"
else
    IMAGE_CONFIG="${SMOKE_IMAGE_CONFIG:-training/_image_test}"
    VIDEO_CONFIG="${SMOKE_VIDEO_CONFIG:-training/_video_test}"
    VALIDATE_CONFIG="${SMOKE_VALIDATE_CONFIG:-configs/efficienttam/efficienttam_ti_512x512.yaml}"
fi

print_image_data_help() {
    echo "[smoke_test] Set DATA_ROOT_IMAGE to a prepared SA-1B-style root, or set DATA_ROOT with a sa1b/ child." >&2
    echo "[smoke_test] Expected layout:" >&2
    echo "[smoke_test]   <image_root>/images/*.jpg" >&2
    echo "[smoke_test]   <image_root>/masks/*.png" >&2
    if [ -n "${DATA_ROOT:-}" ] && [ -d "$DATA_ROOT/raw/sa1b_archives" ]; then
        echo "[smoke_test] Found SA-1B archives at $DATA_ROOT/raw/sa1b_archives." >&2
        echo "[smoke_test] Prepare them with:" >&2
        echo "[smoke_test]   DOWNLOAD_DAVIS=0 DOWNLOAD_EVAL=0 DOWNLOAD_SAV=0 DOWNLOAD_SA1B=1 ./download.sh" >&2
    fi
    echo "[smoke_test] For an already converted sample, run:" >&2
    echo "[smoke_test]   DATA_ROOT_IMAGE=/path/to/sa1b ./smoke_test.sh" >&2
}

print_video_data_help() {
    echo "[smoke_test] Set DATA_ROOT_VIDEO to a prepared SA-V-style root, or set DATA_ROOT with a sav/ child." >&2
    echo "[smoke_test] Expected layout:" >&2
    echo "[smoke_test]   <video_root>/JPEGImages/{video_id}/*.jpg" >&2
    echo "[smoke_test]   <video_root>/Annotations/{video_id}/*.png" >&2
    if [ -n "${DATA_ROOT:-}" ] && [ -d "$DATA_ROOT/raw/sav_sample_extracted" ]; then
        echo "[smoke_test] Found extracted SA-V sample raw data at $DATA_ROOT/raw/sav_sample_extracted." >&2
        echo "[smoke_test] Convert one sample with:" >&2
        echo "[smoke_test]   $PYTHON data/prepare_sav.py --input-dir $DATA_ROOT/raw/sav_sample_extracted --output-dir $DATA_ROOT/sav_sample --annotation-kind both --annotation-stride 4 --max-videos 1" >&2
        echo "[smoke_test] Then rerun:" >&2
        echo "[smoke_test]   ./smoke_test.sh" >&2
    fi
}

if [ -z "$IMAGE_ROOT" ] || [ ! -d "$IMAGE_ROOT/images" ] || [ ! -d "$IMAGE_ROOT/masks" ]; then
    echo "[smoke_test] ERROR: prepared image data is missing images/ or masks/." >&2
    echo "[smoke_test] Resolved image root: ${IMAGE_ROOT:-<empty>}" >&2
    print_image_data_help
    exit 1
fi

if [ -z "$VIDEO_ROOT" ] || [ ! -d "$VIDEO_ROOT/JPEGImages" ] || [ ! -d "$VIDEO_ROOT/Annotations" ]; then
    echo "[smoke_test] ERROR: prepared video data is missing JPEGImages/ or Annotations/." >&2
    echo "[smoke_test] Resolved video root: ${VIDEO_ROOT:-<empty>}" >&2
    print_video_data_help
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
    echo "[smoke_test] ERROR: $name='$root' is not a supported VOS validation layout." >&2
    exit 1
}

if [ "$SMOKE_VALIDATE" = "1" ]; then
    if [ -z "$VAL_ROOT_DAVIS_RESOLVED" ] \
        && [ -z "$VAL_ROOT_MOSE_RESOLVED" ] \
        && [ -z "$VAL_ROOT_LVOS_RESOLVED" ] \
        && [ -z "$VAL_ROOT_SAV_RESOLVED" ] \
        && [ -z "$VAL_ROOT_YTVOS_RESOLVED" ]; then
        echo "[smoke_test] ERROR: set at least one VOS validation root, or use SMOKE_VALIDATE=0." >&2
        echo "[smoke_test] Supported env vars: VAL_ROOT_DAVIS, VAL_ROOT_MOSE, VAL_ROOT_LVOS, VAL_ROOT_SAV, VAL_ROOT_YTVOS." >&2
        exit 1
    fi
    check_vos_root "VAL_ROOT_DAVIS" "$VAL_ROOT_DAVIS_RESOLVED"
    check_vos_root "VAL_ROOT_MOSE" "$VAL_ROOT_MOSE_RESOLVED"
    check_vos_root "VAL_ROOT_LVOS" "$VAL_ROOT_LVOS_RESOLVED"
    check_vos_root "VAL_ROOT_SAV" "$VAL_ROOT_SAV_RESOLVED"
    check_vos_root "VAL_ROOT_YTVOS" "$VAL_ROOT_YTVOS_RESOLVED"
fi

if [ "$SMOKE_IMAGE_EVAL" = "1" ] && [ -n "${SA23_ROOT:-}" ] \
    && { [ ! -d "$SA23_ROOT/images" ] || [ ! -d "$SA23_ROOT/masks" ]; }; then
    echo "[smoke_test] ERROR: SA23_ROOT='$SA23_ROOT' is missing images/ or masks/." >&2
    exit 1
fi

echo "========================================================="
echo "[smoke_test] prepared-data training and evaluation"
echo "[smoke_test] variant=$VARIANT"
echo "[smoke_test] use_paper_configs=$SMOKE_USE_PAPER_CONFIGS"
echo "[smoke_test] image_config=$IMAGE_CONFIG"
echo "[smoke_test] video_config=$VIDEO_CONFIG"
echo "[smoke_test] validate_config=$VALIDATE_CONFIG"
echo "[smoke_test] image_data=$IMAGE_ROOT"
echo "[smoke_test] video_data=$VIDEO_ROOT"
echo "[smoke_test] image_out=$IMAGE_OUT"
echo "[smoke_test] video_out=$VIDEO_OUT"
echo "[smoke_test] eval_dir=$EVAL_DIR"
echo "[smoke_test] nproc=$NPROC_PER_NODE cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[smoke_test] batch_size=$SMOKE_BATCH_SIZE epochs=$SMOKE_EPOCHS max_steps=$SMOKE_MAX_STEPS image_size=$SMOKE_IMAGE_SIZE clip_len=$SMOKE_CLIP_LEN"
echo "[smoke_test] max_val_videos=$SMOKE_MAX_VAL_VIDEOS max_val_images=$SMOKE_MAX_VAL_IMAGES"
echo "[smoke_test] python=$PYTHON"
echo "[smoke_test] validate=$SMOKE_VALIDATE image_eval=$SMOKE_IMAGE_EVAL"
echo "========================================================="

run_module() {
    module_name="$1"
    shift
    if [ "$NPROC_PER_NODE" = "1" ]; then
        "$PYTHON" -m "$module_name" "$@"
        return
    fi
    if ! command -v torchrun >/dev/null 2>&1; then
        echo "[smoke_test] ERROR: torchrun not found. Install PyTorch or use NPROC_PER_NODE=1." >&2
        exit 127
    fi
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" --module "$module_name" "$@"
}

rm -rf "$IMAGE_OUT" "$VIDEO_OUT"

echo "[smoke_test] stage 1 image training"
run_module training.train_image \
    --config "$IMAGE_CONFIG" \
    --data-root "$IMAGE_ROOT" \
    --output-dir "$IMAGE_OUT" \
    --max-steps "$SMOKE_MAX_STEPS" \
    --precision "${SMOKE_PRECISION:-fp32}" \
    train.batch_size="$SMOKE_BATCH_SIZE" \
    train.accumulation_steps=1 \
    train.epochs="$SMOKE_EPOCHS" \
    train.num_workers="${SMOKE_NUM_WORKERS:-0}" \
    train.objects_per_image="${SMOKE_OBJECTS_PER_IMAGE:-1}" \
    train.image_size="$SMOKE_IMAGE_SIZE" \
    train.log_every=1

if [ ! -f "$IMAGE_ARTIFACT" ] || [ ! -f "$IMAGE_OUT/image_final.pt" ]; then
    echo "[smoke_test] ERROR: stage 1 did not produce expected artifact/checkpoint." >&2
    echo "[smoke_test] Expected: $IMAGE_ARTIFACT" >&2
    echo "[smoke_test] Expected: $IMAGE_OUT/image_final.pt" >&2
    exit 1
fi

echo "[smoke_test] stage 2 video training"
run_module training.train_video \
    --config "$VIDEO_CONFIG" \
    --data-root "$VIDEO_ROOT" \
    --image-data-root "$IMAGE_ROOT" \
    --output-dir "$VIDEO_OUT" \
    --init-from "$IMAGE_OUT/image_final.pt" \
    --max-steps "$SMOKE_MAX_STEPS" \
    --precision "${SMOKE_PRECISION:-fp32}" \
    train.batch_size="$SMOKE_BATCH_SIZE" \
    train.accumulation_steps=1 \
    train.epochs="$SMOKE_EPOCHS" \
    train.num_workers="${SMOKE_NUM_WORKERS:-0}" \
    train.clip_len="$SMOKE_CLIP_LEN" \
    train.stride_choices="[1]" \
    train.objects_per_clip="${SMOKE_OBJECTS_PER_CLIP:-1}" \
    train.image_mix_prob="${SMOKE_IMAGE_MIX_PROB:-0.1}" \
    train.image_size="$SMOKE_IMAGE_SIZE" \
    train.log_every=1

if [ ! -f "$VIDEO_ARTIFACT" ] || [ ! -f "$VIDEO_OUT/video_final.pt" ]; then
    echo "[smoke_test] ERROR: stage 2 did not produce expected artifact/checkpoint." >&2
    echo "[smoke_test] Expected: $VIDEO_ARTIFACT" >&2
    echo "[smoke_test] Expected: $VIDEO_OUT/video_final.pt" >&2
    exit 1
fi

if [ "$SMOKE_VALIDATE" = "1" ]; then
    echo "[smoke_test] stage 3 VOS benchmark evaluation"
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

    val_args=()
    if [ -n "$SMOKE_MAX_VAL_VIDEOS" ]; then
        val_args+=(--max-videos "$SMOKE_MAX_VAL_VIDEOS")
    fi
    # Smoke runs are short; skip the slow max-autotune image-encoder compile.
    # Set SMOKE_VALIDATE_COMPILE=1 to keep compilation on.
    if [ "${SMOKE_VALIDATE_COMPILE:-0}" != "1" ]; then
        val_args+=(--no-compile)
    fi
    "$PYTHON" -m tools.validate_vos_suite \
        --config "$VALIDATE_CONFIG" \
        --ckpt "$VIDEO_OUT/video_final.pt" \
        --output-json "$EVAL_DIR/vos_suite.json" \
        "${val_args[@]}"
else
    echo "[smoke_test] skipping VOS validation because SMOKE_VALIDATE=$SMOKE_VALIDATE"
fi

if [ "$SMOKE_IMAGE_EVAL" = "1" ] && [ -n "${SA23_ROOT:-}" ]; then
    echo "[smoke_test] stage 4 SA-23 image mIoU evaluation"
    image_eval_args=()
    if [ -n "$SMOKE_MAX_VAL_IMAGES" ]; then
        image_eval_args+=(--max-images "$SMOKE_MAX_VAL_IMAGES")
    fi
    if [ "${SMOKE_VALIDATE_COMPILE:-0}" != "1" ]; then
        image_eval_args+=(--no-compile)
    fi
    "$PYTHON" -m tools.validate_image_miou \
        --config "$VALIDATE_CONFIG" \
        --ckpt "$VIDEO_OUT/video_final.pt" \
        --root "$SA23_ROOT" \
        --output-json "$EVAL_DIR/sa23_miou.json" \
        "${image_eval_args[@]}"
else
    echo "[smoke_test] skipping SA-23 image mIoU because SMOKE_IMAGE_EVAL=$SMOKE_IMAGE_EVAL or SA23_ROOT is not set."
fi

echo "========================================================="
echo "[smoke_test] done"
echo "[smoke_test] image checkpoint: $IMAGE_OUT/image_final.pt"
echo "[smoke_test] video checkpoint: $VIDEO_OUT/video_final.pt"
echo "[smoke_test] evaluation dir: $EVAL_DIR"
echo "========================================================="
