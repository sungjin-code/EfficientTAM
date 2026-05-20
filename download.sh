#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/datasets}"
SA1B_DIR="${DATA_ROOT_IMAGE:-$DATA_DIR/SA1B}"
SAV_DIR="${DATA_ROOT_VIDEO:-$DATA_DIR/SA-V}"
DAVIS_DIR="${VAL_ROOT_VIDEO:-$DATA_DIR/DAVIS}"
DOWNLOAD_DAVIS="${DOWNLOAD_DAVIS:-1}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$DATA_DIR"

echo "========================================================="
echo "EfficientTAM paper-data setup check"
echo "DATA_DIR=$DATA_DIR"
echo "SA-1B prepared root=$SA1B_DIR"
echo "SA-V prepared root=$SAV_DIR"
echo "DAVIS validation root=$DAVIS_DIR"
echo "PYTHON=$PYTHON"
echo "========================================================="

have_stage1=false
have_stage2=false
have_val=false

if [ -d "$SA1B_DIR/images" ] && [ -d "$SA1B_DIR/masks" ]; then
    echo "[1/3] SA-1B-style image data already prepared."
    have_stage1=true
elif [ -n "${SA1B_RAW_DIR:-}" ]; then
    echo "[1/3] Converting SA-1B raw dump from $SA1B_RAW_DIR"
    $PYTHON "$ROOT_DIR/data/prepare_sa1b.py" \
        --input_dir "$SA1B_RAW_DIR" \
        --output_dir "$SA1B_DIR"
    if [ -d "$SA1B_DIR/images" ] && [ -d "$SA1B_DIR/masks" ]; then
        have_stage1=true
    fi
else
    echo "[1/3] Missing SA-1B-style image data."
    echo "      Download SA-1B from Meta after accepting its terms, extract it,"
    echo "      then rerun with SA1B_RAW_DIR=/path/to/extracted_sa1b."
fi

if [ -d "$SAV_DIR/JPEGImages" ] && [ -d "$SAV_DIR/Annotations" ]; then
    echo "[2/3] SA-V-style video data already prepared."
    have_stage2=true
elif [ -n "${SAV_RAW_DIR:-}" ]; then
    echo "[2/3] Converting SA-V raw dump from $SAV_RAW_DIR"
    $PYTHON "$ROOT_DIR/data/prepare_sav.py" \
        --input-dir "$SAV_RAW_DIR" \
        --output-dir "$SAV_DIR" \
        --annotation-kind "${SAV_ANNOTATION_KIND:-both}" \
        --annotation-stride "${SAV_ANNOTATION_STRIDE:-4}"
    if [ -d "$SAV_DIR/JPEGImages" ] && [ -d "$SAV_DIR/Annotations" ]; then
        have_stage2=true
    fi
else
    echo "[2/3] Missing SA-V-style video data."
    echo "      Download SA-V from Meta, extract it, then rerun with:"
    echo "      SAV_RAW_DIR=/path/to/extracted_sav"
    echo "      The converted output will use this layout:"
    echo "      $SAV_DIR/JPEGImages/{video_id}/00000.jpg"
    echo "      $SAV_DIR/Annotations/{video_id}/00000.png"
    echo "      DAVIS is not a paper-equivalent replacement for SA-V training."
fi

if [ "$DOWNLOAD_DAVIS" = "1" ]; then
    if [ -d "$DAVIS_DIR/JPEGImages" ] && [ -d "$DAVIS_DIR/Annotations" ]; then
        echo "[3/3] DAVIS validation data already exists."
        have_val=true
    else
        echo "[3/3] Downloading DAVIS 2017 TrainVal 480p for validation."
        tmp_zip="$DATA_DIR/DAVIS-2017-trainval-480p.zip"
        if command -v wget >/dev/null 2>&1; then
            wget -c \
                "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip" \
                -O "$tmp_zip"
        elif command -v curl >/dev/null 2>&1; then
            curl -L \
                "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip" \
                -o "$tmp_zip"
        else
            echo "Error: install wget or curl to download DAVIS." >&2
            exit 1
        fi
        unzip -q "$tmp_zip" -d "$DATA_DIR"
        rm "$tmp_zip"
        if [ "$DAVIS_DIR" != "$DATA_DIR/DAVIS" ] && [ -d "$DATA_DIR/DAVIS" ]; then
            mkdir -p "$(dirname "$DAVIS_DIR")"
            mv "$DATA_DIR/DAVIS" "$DAVIS_DIR"
        fi
        have_val=true
    fi
else
    echo "[3/3] Skipping DAVIS download because DOWNLOAD_DAVIS=$DOWNLOAD_DAVIS."
fi

echo ""
echo "Recommended .env values:"
echo "DATA_ROOT_IMAGE=$SA1B_DIR"
echo "DATA_ROOT_VIDEO=$SAV_DIR"
echo "VAL_ROOT_DAVIS=$DAVIS_DIR"
echo "# Optional paper metric roots:"
echo "# VAL_ROOT_MOSE=/path/to/mose_val_root"
echo "# VAL_ROOT_LVOS=/path/to/lvos_val_root"
echo "# VAL_ROOT_SAV=/path/to/sav_test_or_val_root"
echo "# VAL_ROOT_YTVOS=/path/to/ytvos2019_val_root"
echo "# SA23_ROOT=/path/to/sa23_style_root"

if [ "$have_stage1" != "true" ] || [ "$have_stage2" != "true" ]; then
    echo ""
    echo "Paper training data is not fully ready yet."
    echo "This script no longer creates dummy SA-1B data because that would make"
    echo "a training run look valid while using non-paper data."
    exit 1
fi

echo ""
echo "Paper training data roots are ready."
if [ "$have_val" = "true" ]; then
    echo "DAVIS validation root is ready."
fi
