#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then
    declare -A _ENV_OVERRIDES=()
    while IFS= read -r _name; do
        _ENV_OVERRIDES["$_name"]="${!_name}"
    done < <(compgen -e)
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
    for _name in "${!_ENV_OVERRIDES[@]}"; do
        export "$_name=${_ENV_OVERRIDES[$_name]}"
    done
    unset _name _ENV_OVERRIDES
fi

DATA_DIR="${DATA_ROOT:-${DATA_DIR:-$ROOT_DIR/datasets}}"
SA1B_DIR="${DATA_ROOT_IMAGE:-$DATA_DIR/sa1b}"
SAV_DIR="${DATA_ROOT_VIDEO:-$DATA_DIR/sav}"
DAVIS_DIR="${VAL_ROOT_DAVIS:-${VAL_ROOT_VIDEO:-$DATA_DIR/DAVIS}}"
MOSE_DIR="${VAL_ROOT_MOSE:-$DATA_DIR/MOSE}"
LVOS_DIR="${VAL_ROOT_LVOS:-$DATA_DIR/LVOS}"
SAV_TEST_DIR="${VAL_ROOT_SAV:-$DATA_DIR/SA-V-test}"
YTVOS_DIR="${VAL_ROOT_YTVOS:-$DATA_DIR/YTVOS2019}"
RAW_DIR="${RAW_DATA_DIR:-$DATA_DIR/raw}"
SA1B_MANIFEST="${SA1B_MANIFEST:-$ROOT_DIR/data/SA-1B.txt}"
SAV_MANIFEST="${SAV_MANIFEST:-$ROOT_DIR/data/SA-V.txt}"
DOWNLOAD_DAVIS="${DOWNLOAD_DAVIS:-1}"
DOWNLOAD_EVAL="${DOWNLOAD_EVAL:-1}"
DOWNLOAD_MOSE="${DOWNLOAD_MOSE:-$DOWNLOAD_EVAL}"
DOWNLOAD_LVOS="${DOWNLOAD_LVOS:-$DOWNLOAD_EVAL}"
DOWNLOAD_SAV_TEST="${DOWNLOAD_SAV_TEST:-$DOWNLOAD_EVAL}"
DOWNLOAD_YTVOS="${DOWNLOAD_YTVOS:-$DOWNLOAD_EVAL}"
DOWNLOAD_SA1B="${DOWNLOAD_SA1B:-1}"
DOWNLOAD_SAV="${DOWNLOAD_SAV:-1}"
EXTRACT_DATASETS="${EXTRACT_DATASETS:-1}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-1}"
DOWNLOAD_JOBS="${DOWNLOAD_JOBS:-1}"
PYTHON="${PYTHON:-python3}"
MOSE_HF_REPO="${MOSE_HF_REPO:-FudanCVL/MOSEv2}"
LVOS_GDRIVE_URL="${LVOS_GDRIVE_URL:-https://drive.google.com/file/d/1-ehpl5s0Fd14WwtT-GmWtIWa_BxZl9D6/view}"
YTVOS_GDRIVE_URL="${YTVOS_GDRIVE_URL:-https://drive.google.com/drive/folders/1XwjQ-eysmOb7JdmJAwfVOBZX-aMbHccC}"

mkdir -p "$DATA_DIR"
mkdir -p "$RAW_DIR"

echo "========================================================="
echo "EfficientTAM paper-data setup check"
echo "DATA_DIR=$DATA_DIR"
echo "RAW_DIR=$RAW_DIR"
echo "SA-1B prepared root=$SA1B_DIR"
echo "SA-V prepared root=$SAV_DIR"
echo "DAVIS validation root=$DAVIS_DIR"
echo "MOSE validation root=$MOSE_DIR"
echo "LVOS validation root=$LVOS_DIR"
echo "SA-V test root=$SAV_TEST_DIR"
echo "YTVOS validation root=$YTVOS_DIR"
echo "PYTHON=$PYTHON"
echo "========================================================="

have_stage1=false
have_stage2=false
have_val=false
failed_downloads=()

image_layout_ready() {
    [ -d "$1/images" ] && [ -d "$1/masks" ] \
        && [ -n "$(find "$1/images" -maxdepth 1 -type f -print -quit)" ] \
        && [ -n "$(find "$1/masks" -maxdepth 1 -type f -print -quit)" ]
}

video_layout_ready() {
    [ -d "$1/JPEGImages" ] && [ -d "$1/Annotations" ] \
        && [ -n "$(find "$1/JPEGImages" -mindepth 1 -maxdepth 1 -type d -print -quit)" ] \
        && [ -n "$(find "$1/Annotations" -mindepth 1 -maxdepth 1 -type d -print -quit)" ]
}

davis_layout_ready() {
    [ -d "$1/JPEGImages" ] && [ -d "$1/Annotations" ]
}

vos_layout_ready() {
    local root="$1"
    { [ -d "$root/JPEGImages" ] && [ -d "$root/Annotations" ]; } \
        || { [ -d "$root/JPEGImages/480p" ] && [ -d "$root/Annotations/480p" ]; } \
        || { [ -d "$root/JPEGImages_24fps" ] && [ -d "$root/Annotations_6fps" ]; } \
        || { [ -d "$root/valid/JPEGImages" ] && [ -d "$root/valid/Annotations" ]; } \
        || { [ -d "$root/val/JPEGImages" ] && [ -d "$root/val/Annotations" ]; } \
        || { [ -d "$root/test/JPEGImages" ] && [ -d "$root/test/Annotations" ]; } \
        || { [ -d "$root/train/JPEGImages" ] && [ -d "$root/train/Annotations" ]; } \
        || { [ -d "$root/valid/JPEGImages_24fps" ] && [ -d "$root/valid/Annotations_6fps" ]; } \
        || { [ -d "$root/val/JPEGImages_24fps" ] && [ -d "$root/val/Annotations_6fps" ]; } \
        || { [ -d "$root/test/JPEGImages_24fps" ] && [ -d "$root/test/Annotations_6fps" ]; } \
        || { [ -d "$root/train/JPEGImages_24fps" ] && [ -d "$root/train/Annotations_6fps" ]; } \
        || { [ -d "$root/sav_test/JPEGImages_24fps" ] && [ -d "$root/sav_test/Annotations_6fps" ]; }
}

download_one() {
    local url="$1"
    local output="$2"
    mkdir -p "$(dirname "$output")"
    if [ -s "$output" ]; then
        echo "      exists: $(basename "$output")"
        return
    fi
    if command -v wget >/dev/null 2>&1; then
        wget -c "$url" -O "$output"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -C - "$url" -o "$output"
    else
        echo "Error: install wget or curl to download datasets." >&2
        exit 1
    fi
}

download_manifest() {
    local manifest="$1"
    local archive_dir="$2"
    local pattern="$3"
    local limit="${4:-0}"

    if [ ! -f "$manifest" ]; then
        echo "Error: manifest not found: $manifest" >&2
        echo "  Download the manifest file from the dataset's official page and place it in $ROOT_DIR/data/" >&2
        exit 1
    fi
    mkdir -p "$archive_dir"

    awk -v pattern="$pattern" -v limit="$limit" '
        NR == 1 && $1 == "file_name" { next }
        $1 ~ pattern {
            print $1 "\t" $2
            count++
            if (limit > 0 && count >= limit) exit
        }
    ' "$manifest" > "$archive_dir/.download_entries.tsv"

    if [ ! -s "$archive_dir/.download_entries.tsv" ]; then
        echo "Error: no entries matched '$pattern' in $manifest" >&2
        exit 1
    fi

    if [ "$DOWNLOAD_JOBS" -gt 1 ] && command -v xargs >/dev/null 2>&1; then
        while IFS="$(printf '\t')" read -r file_name url; do
            printf '%s\t%s\t%s\n' "$url" "$archive_dir/$file_name" "$file_name"
        done < "$archive_dir/.download_entries.tsv" | xargs -n 3 -P "$DOWNLOAD_JOBS" sh -c '
            url="$1"
            output="$2"
            file_name="$3"
            if [ -s "$output" ]; then
                echo "      exists: $file_name"
                exit 0
            fi
            mkdir -p "$(dirname "$output")"
            if command -v wget >/dev/null 2>&1; then
                wget -c "$url" -O "$output"
            else
                curl -L -C - "$url" -o "$output"
            fi
        ' sh
    else
        while IFS="$(printf '\t')" read -r file_name url; do
            output="$archive_dir/$file_name"
            if [ -s "$output" ]; then
                echo "      exists: $file_name"
            else
                echo "      downloading: $file_name"
                download_one "$url" "$output"
            fi
        done < "$archive_dir/.download_entries.tsv"
    fi
}

extract_tars() {
    local archive_dir="$1"
    local extract_dir="$2"
    mkdir -p "$extract_dir"
    find "$archive_dir" -maxdepth 1 -type f -name "*.tar" | sort | while read -r tar_path; do
        local shard_name
        shard_name="$(basename "$tar_path" .tar)"
        local stamp="$extract_dir/.extracted_${shard_name}.stamp"
        if [ -f "$stamp" ]; then
            echo "      extracted: $(basename "$tar_path")"
            continue
        fi
        echo "      extracting: $(basename "$tar_path")"
        mkdir -p "$extract_dir/$shard_name"
        tar -xf "$tar_path" -C "$extract_dir/$shard_name"
        touch "$stamp"
        if [ "$KEEP_ARCHIVES" != "1" ]; then
            rm "$tar_path"
        fi
    done
}

extract_tars_flat() {
    local archive_dir="$1"
    local extract_dir="$2"
    mkdir -p "$extract_dir"
    find "$archive_dir" -maxdepth 1 -type f -name "*.tar" | sort | while read -r tar_path; do
        local shard_name
        shard_name="$(basename "$tar_path" .tar)"
        local stamp="$extract_dir/.extracted_${shard_name}.stamp"
        if [ -f "$stamp" ]; then
            echo "      extracted: $(basename "$tar_path")"
            continue
        fi
        echo "      extracting: $(basename "$tar_path")"
        tar -xf "$tar_path" -C "$extract_dir"
        touch "$stamp"
        if [ "$KEEP_ARCHIVES" != "1" ]; then
            rm "$tar_path"
        fi
    done
}

extract_and_convert_sa1b() {
    local archive_dir="$1"
    local extract_dir="$2"
    local output_dir="$3"
    local tmp_shard="$extract_dir/_shard_tmp"

    find "$archive_dir" -maxdepth 1 -type f -name "*.tar" | sort | while read -r tar_path; do
        local shard_name
        shard_name="$(basename "$tar_path" .tar)"
        local stamp="$output_dir/.converted_${shard_name}.stamp"
        if [ -f "$stamp" ]; then
            echo "      already converted: $(basename "$tar_path")"
            continue
        fi
        echo "      extracting: $(basename "$tar_path")"
        rm -rf "$tmp_shard"
        mkdir -p "$tmp_shard"
        tar -xf "$tar_path" -C "$tmp_shard"

        echo "      converting shard: $shard_name"
        $PYTHON "$ROOT_DIR/data/prepare_sa1b.py" \
            --input_dir "$tmp_shard" \
            --output_dir "$output_dir" \
            --progress_every 0

        rm -rf "$tmp_shard"
        touch "$stamp"
        if [ "$KEEP_ARCHIVES" != "1" ]; then
            rm "$tar_path"
        fi
    done
}

extract_archives_in_place() {
    local target_dir="$1"
    find "$target_dir" -maxdepth 2 -type f \( \
        -name "*.zip" -o -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" \
    \) | sort | while read -r archive_path; do
        local stamp="${archive_path}.extracted.stamp"
        if [ -f "$stamp" ]; then
            echo "      extracted: $(basename "$archive_path")"
            continue
        fi
        echo "      extracting: $(basename "$archive_path")"
        case "$archive_path" in
            *.zip)
                unzip -q "$archive_path" -d "$target_dir"
                ;;
            *.tar|*.tar.gz|*.tgz)
                tar -xf "$archive_path" -C "$target_dir"
                ;;
        esac
        touch "$stamp"
    done
}

download_hf_dataset() {
    local repo="$1"
    local target_dir="$2"
    shift 2
    mkdir -p "$target_dir"
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$repo" \
            --repo-type dataset \
            --local-dir "$target_dir" \
            "$@"
    else
        echo "Error: huggingface-cli is required for $repo." >&2
        echo "Install dependencies with: pip install -r requirements.txt" >&2
        exit 1
    fi
}

download_gdrive_file() {
    local url="$1"
    local output="$2"
    local file_id="$url"
    mkdir -p "$(dirname "$output")"
    if [ -s "$output" ]; then
        echo "      exists: $(basename "$output")"
        return
    fi
    if ! command -v gdown >/dev/null 2>&1; then
        echo "Error: gdown is required for Google Drive downloads." >&2
        echo "Install dependencies with: pip install -r requirements.txt" >&2
        exit 1
    fi
    case "$url" in
        *"/file/d/"*)
            file_id="${url#*/file/d/}"
            file_id="${file_id%%/*}"
            ;;
        *"id="*)
            file_id="${url#*id=}"
            file_id="${file_id%%&*}"
            ;;
    esac
    local gdrive_url="https://drive.google.com/uc?id=$file_id"
    [ "$file_id" = "$url" ] && gdrive_url="$url"

    local cookie_args=()
    if [ -n "${GDRIVE_COOKIES:-}" ] && [ -f "$GDRIVE_COOKIES" ]; then
        cookie_args=(--cookies "$GDRIVE_COOKIES")
    fi

    local fuzzy_args=()
    if gdown --help 2>/dev/null | grep -q -- "--fuzzy"; then
        fuzzy_args=(--fuzzy)
    fi

    if gdown "${fuzzy_args[@]}" "${cookie_args[@]}" "$gdrive_url" -O "$output"; then
        return 0
    fi

    echo "" >&2
    echo "      Google Drive download failed for: $(basename "$output")" >&2
    echo "      To fix, choose one of:" >&2
    echo "        A) Download manually from browser and save to:" >&2
    echo "             $output" >&2
    echo "        B) Export cookies from a logged-in Chrome/Firefox session:" >&2
    echo "             pip install browser-cookie3  # or use 'Get cookies.txt' extension" >&2
    echo "             GDRIVE_COOKIES=/path/to/cookies.txt bash download.sh" >&2
    return 1
}

download_gdrive_folder() {
    local url="$1"
    local output_dir="$2"
    local remaining_ok=()
    mkdir -p "$output_dir"
    if ! command -v gdown >/dev/null 2>&1; then
        echo "Error: gdown is required for Google Drive folder downloads." >&2
        echo "Install dependencies with: pip install -r requirements.txt" >&2
        exit 1
    fi
    if gdown --help 2>/dev/null | grep -q -- "--remaining-ok"; then
        remaining_ok=(--remaining-ok)
    fi
    gdown --folder "$url" -O "$output_dir" "${remaining_ok[@]}"
}

download_davis() {
    if davis_layout_ready "$DAVIS_DIR"; then
        echo "[1/3] DAVIS validation data already exists."
        have_val=true
        return
    fi

    echo "[1/3] Downloading DAVIS 2017 TrainVal 480p."
    local davis_archive_dir="$RAW_DIR/davis_archives"
    local tmp_zip="$davis_archive_dir/DAVIS-2017-trainval-480p.zip"
    mkdir -p "$davis_archive_dir"
    download_one \
        "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip" \
        "$tmp_zip"
    unzip -q -n "$tmp_zip" -d "$DATA_DIR"
    if [ "$KEEP_ARCHIVES" != "1" ]; then
        rm "$tmp_zip"
    fi
    if [ "$DAVIS_DIR" != "$DATA_DIR/DAVIS" ] && [ -d "$DATA_DIR/DAVIS" ]; then
        mkdir -p "$(dirname "$DAVIS_DIR")"
        mv "$DATA_DIR/DAVIS" "$DAVIS_DIR"
    fi
    have_val=true
}

if [ "$DOWNLOAD_DAVIS" = "1" ]; then
    if ! download_davis; then
        failed_downloads+=("DAVIS [https://data.vision.ee.ethz.ch/csergi/share/davis/]")
        echo "[warn] DAVIS download failed, skipping." >&2
    fi
else
    echo "[1/3] Skipping DAVIS download because DOWNLOAD_DAVIS=$DOWNLOAD_DAVIS."
    if davis_layout_ready "$DAVIS_DIR"; then
        have_val=true
    fi
fi

if [ "$DOWNLOAD_MOSE" = "1" ]; then
    if vos_layout_ready "$MOSE_DIR"; then
        echo "[eval] MOSE validation data already exists."
    else
        echo "[eval] Downloading MOSE validation data from Hugging Face."
        if ! (
            set -euo pipefail
            download_hf_dataset "$MOSE_HF_REPO" "$MOSE_DIR" --include "*valid*"
            extract_archives_in_place "$MOSE_DIR"
        ); then
            failed_downloads+=("MOSE [HuggingFace: $MOSE_HF_REPO]")
            echo "[warn] MOSE download failed, skipping." >&2
        fi
    fi
fi

if [ "$DOWNLOAD_LVOS" = "1" ]; then
    if vos_layout_ready "$LVOS_DIR"; then
        echo "[eval] LVOS validation data already exists."
    else
        echo "[eval] Downloading LVOS validation data from Google Drive."
        if ! (
            set -euo pipefail
            _archive_dir="$RAW_DIR/lvos_archives"
            mkdir -p "$_archive_dir"
            download_gdrive_file "$LVOS_GDRIVE_URL" "$_archive_dir/lvos_val.zip" || exit 1
            mkdir -p "$LVOS_DIR"
            unzip -q -n "$_archive_dir/lvos_val.zip" -d "$LVOS_DIR"
            if [ -d "$LVOS_DIR/LVOS" ] && [ ! -d "$LVOS_DIR/JPEGImages" ]; then
                find "$LVOS_DIR/LVOS" -mindepth 1 -maxdepth 1 -exec mv {} "$LVOS_DIR" \;
            elif [ -d "$LVOS_DIR/lvos" ] && [ ! -d "$LVOS_DIR/JPEGImages" ]; then
                find "$LVOS_DIR/lvos" -mindepth 1 -maxdepth 1 -exec mv {} "$LVOS_DIR" \;
            fi
        ); then
            failed_downloads+=("LVOS [Google Drive: $LVOS_GDRIVE_URL]")
            echo "[warn] LVOS download failed, skipping." >&2
        fi
    fi
fi

if [ "$DOWNLOAD_SAV_TEST" = "1" ]; then
    if vos_layout_ready "$SAV_TEST_DIR"; then
        echo "[eval] SA-V test data already exists."
    else
        echo "[eval] Downloading SA-V test data from $SAV_MANIFEST"
        if ! (
            set -euo pipefail
            _archive_dir="$RAW_DIR/sav_eval_archives"
            download_manifest "$SAV_MANIFEST" "$_archive_dir" '^sav_test[.]tar$' 0
            extract_tars_flat "$_archive_dir" "$SAV_TEST_DIR"
        ); then
            failed_downloads+=("SA-V test [$SAV_MANIFEST]")
            echo "[warn] SA-V test download failed, skipping." >&2
        fi
    fi
fi

if [ "$DOWNLOAD_YTVOS" = "1" ]; then
    if vos_layout_ready "$YTVOS_DIR"; then
        echo "[eval] YTVOS 2019 validation data already exists."
    else
        echo "[eval] Downloading YTVOS 2019 validation data from Google Drive."
        if ! (
            set -euo pipefail
            download_gdrive_folder "$YTVOS_GDRIVE_URL" "$YTVOS_DIR"
            extract_archives_in_place "$YTVOS_DIR"
        ); then
            failed_downloads+=("YTVOS 2019 [Google Drive: $YTVOS_GDRIVE_URL]")
            echo "[warn] YTVOS 2019 download failed, skipping." >&2
        fi
    fi
fi

if image_layout_ready "$SA1B_DIR"; then
    echo "[2/3] SA-1B-style image data already prepared."
    have_stage1=true
elif [ -n "${SA1B_RAW_DIR:-}" ]; then
    echo "[2/3] Converting SA-1B raw dump from $SA1B_RAW_DIR"
    if ! (
        set -euo pipefail
        $PYTHON "$ROOT_DIR/data/prepare_sa1b.py" \
            --input_dir "$SA1B_RAW_DIR" \
            --output_dir "$SA1B_DIR"
    ); then
        failed_downloads+=("SA-1B conversion from $SA1B_RAW_DIR")
        echo "[warn] SA-1B conversion failed, skipping." >&2
    fi
    if image_layout_ready "$SA1B_DIR"; then
        have_stage1=true
    fi
elif [ "$DOWNLOAD_SA1B" = "1" ]; then
    echo "[2/3] Downloading and preparing SA-1B from $SA1B_MANIFEST"
    if ! (
        set -euo pipefail
        _archive_dir="$RAW_DIR/sa1b_archives"
        _extract_dir="$RAW_DIR/sa1b_extracted"
        download_manifest "$SA1B_MANIFEST" "$_archive_dir" '^sa_.*[.]tar$' "${SA1B_LIMIT:-0}"
        if [ "$EXTRACT_DATASETS" = "1" ]; then
            mkdir -p "$_extract_dir"
            mkdir -p "$SA1B_DIR"
            echo "      converting SA-1B to $SA1B_DIR (one shard at a time to save disk)"
            extract_and_convert_sa1b "$_archive_dir" "$_extract_dir" "$SA1B_DIR"
        else
            echo "      skipped SA-1B extraction because EXTRACT_DATASETS=$EXTRACT_DATASETS"
        fi
    ); then
        failed_downloads+=("SA-1B [$SA1B_MANIFEST]")
        echo "[warn] SA-1B download failed, skipping." >&2
    fi
    if image_layout_ready "$SA1B_DIR"; then
        have_stage1=true
    fi
else
    echo "[2/3] Missing SA-1B-style image data."
    echo "      Download SA-1B through $SA1B_MANIFEST, or set"
    echo "      SA1B_RAW_DIR=/path/to/extracted_sa1b."
fi

if video_layout_ready "$SAV_DIR"; then
    echo "[3/3] SA-V-style video data already prepared."
    have_stage2=true
elif [ -n "${SAV_RAW_DIR:-}" ]; then
    echo "[3/3] Converting SA-V raw dump from $SAV_RAW_DIR"
    if ! (
        set -euo pipefail
        $PYTHON "$ROOT_DIR/data/prepare_sav.py" \
            --input-dir "$SAV_RAW_DIR" \
            --output-dir "$SAV_DIR" \
            --annotation-kind "${SAV_ANNOTATION_KIND:-both}" \
            --annotation-stride "${SAV_ANNOTATION_STRIDE:-4}"
    ); then
        failed_downloads+=("SA-V conversion from $SAV_RAW_DIR")
        echo "[warn] SA-V conversion failed, skipping." >&2
    fi
    if video_layout_ready "$SAV_DIR"; then
        have_stage2=true
    fi
elif [ "$DOWNLOAD_SAV" = "1" ]; then
    echo "[3/3] Downloading and preparing SA-V from $SAV_MANIFEST"
    if ! (
        set -euo pipefail
        _archive_dir="$RAW_DIR/sav_archives"
        _extract_dir="$RAW_DIR/sav_extracted"
        download_manifest "$SAV_MANIFEST" "$_archive_dir" '^(sav_[0-9][0-9][0-9][.]tar|videos_fps_24[.]tar|videos_fps_6[.]tar)$' "${SAV_LIMIT:-0}"
        download_manifest "$SAV_MANIFEST" "$_archive_dir" '^sav_.*sum[.]chk' 0
        if [ "$EXTRACT_DATASETS" = "1" ]; then
            extract_tars "$_archive_dir" "$_extract_dir"
            echo "      converting SA-V to $SAV_DIR"
            $PYTHON "$ROOT_DIR/data/prepare_sav.py" \
                --input-dir "$_extract_dir" \
                --output-dir "$SAV_DIR" \
                --annotation-kind "${SAV_ANNOTATION_KIND:-both}" \
                --annotation-stride "${SAV_ANNOTATION_STRIDE:-4}"
        else
            echo "      skipped SA-V extraction because EXTRACT_DATASETS=$EXTRACT_DATASETS"
        fi
    ); then
        failed_downloads+=("SA-V [$SAV_MANIFEST]")
        echo "[warn] SA-V download failed, skipping." >&2
    fi
    if video_layout_ready "$SAV_DIR"; then
        have_stage2=true
    fi
else
    echo "[3/3] Missing SA-V-style video data."
    echo "      Download SA-V through $SAV_MANIFEST, or set"
    echo "      SAV_RAW_DIR=/path/to/extracted_sav."
    echo "      The converted output will use this layout:"
    echo "      $SAV_DIR/JPEGImages/{video_id}/00000.jpg"
    echo "      $SAV_DIR/Annotations/{video_id}/00000.png"
fi

if [ "${#failed_downloads[@]}" -gt 0 ]; then
    echo ""
    echo "========= FAILED DOWNLOADS ========="
    for _item in "${failed_downloads[@]}"; do
        echo "  - $_item"
    done
    echo "====================================="
fi

echo ""
echo "Recommended .env values:"
echo "DATA_ROOT_IMAGE=$SA1B_DIR"
echo "DATA_ROOT_VIDEO=$SAV_DIR"
echo "VAL_ROOT_DAVIS=$DAVIS_DIR"
echo "VAL_ROOT_MOSE=$MOSE_DIR"
echo "VAL_ROOT_LVOS=$LVOS_DIR"
echo "VAL_ROOT_SAV=$SAV_TEST_DIR"
echo "VAL_ROOT_YTVOS=$YTVOS_DIR"
echo "RAW_DATA_DIR=$RAW_DIR"
echo "# Optional paper metric roots:"
echo "# VAL_ROOT_MOSE=/path/to/mose_val_root"
echo "# VAL_ROOT_LVOS=/path/to/lvos_val_root"
echo "# VAL_ROOT_SAV=/path/to/sav_test_or_val_root"
echo "# VAL_ROOT_YTVOS=/path/to/ytvos2019_val_root"
echo "# SA23_ROOT=/path/to/sa23_style_root"

if [ "$have_stage1" != "true" ] || [ "$have_stage2" != "true" ]; then
    echo ""
    echo "Paper training data is not fully ready yet."
    echo "Only paper datasets are supported: SA-1B for Stage 1 and SA-V for Stage 2."
    exit 1
fi

echo ""
echo "Training data roots are ready."
if [ "$have_val" = "true" ]; then
    echo "DAVIS validation root is ready."
fi
