# Data Path Contract

This repository uses three distinct data states. Scripts must not treat them as
interchangeable:

1. **Archive cache**: downloaded `.tar`/`.zip` files under `$RAW_DATA_DIR`.
2. **Raw extracted cache**: extracted paper files, still in original dataset
   format, under `$RAW_DATA_DIR`.
3. **Prepared training/evaluation roots**: layouts consumed by training and
   validation code.

With the current `.env`:

```text
DATA_ROOT=./_data
RAW_DATA_DIR defaults to ./_data/raw
```

## Current Local Raw Data

The current `_data/raw` tree contains these important entries:

```text
_data/raw/sa1b_archives/
_data/raw/sa1b_extracted/sa_000028/
_data/raw/sav_extracted/sav_000/
_data/raw/sav_sample_archives/
_data/raw/sa1b_sample_extracted/
_data/raw/sav_sample_extracted/
```

Canonical paths are:

```text
_data/raw/sa1b_archives
_data/raw/sa1b_extracted
_data/raw/sav_archives
_data/raw/sav_extracted
```

The `*_sample_*` paths are legacy/debug paths. New scripts should not depend on
them. If sample data is reused, move or copy it into the canonical paths first.

## Prepared Training Roots

Stage 1 image training requires:

```text
{DATA_ROOT_IMAGE or DATA_ROOT/sa1b}/images/*.jpg
{DATA_ROOT_IMAGE or DATA_ROOT/sa1b}/masks/*.png
```

Stage 2 video training requires:

```text
{DATA_ROOT_VIDEO or DATA_ROOT/sav}/JPEGImages/{video_id}/*.jpg
{DATA_ROOT_VIDEO or DATA_ROOT/sav}/Annotations/{video_id}/*.png
```

`_data/SA-V-test` is not a Stage 2 training root. It is an evaluation root with
SA-V test layout:

```text
_data/SA-V-test/sav_test/JPEGImages_24fps/{video_id}/*.jpg
_data/SA-V-test/sav_test/Annotations_6fps/{video_id}/{object_id}/*.png
```

Use it through `VAL_ROOT_SAV`, not `DATA_ROOT_VIDEO`.

## `download.sh`

`download.sh` is the producer of prepared data. It resolves paths as follows:

```text
DATA_DIR = DATA_ROOT, then DATA_DIR, then ./datasets
RAW_DIR = RAW_DATA_DIR, then DATA_DIR/raw
SA1B_DIR = DATA_ROOT_IMAGE, then DATA_DIR/sa1b
SAV_DIR = DATA_ROOT_VIDEO, then DATA_DIR/sav
```

For training data, it should write only these prepared roots:

```text
$SA1B_DIR/images
$SA1B_DIR/masks
$SAV_DIR/JPEGImages
$SAV_DIR/Annotations
```

For the current raw sample cache, the intended conversion command is:

```bash
DOWNLOAD_DAVIS=0 DOWNLOAD_EVAL=0 \
SA1B_RAW_DIR=./_data/raw/sa1b_extracted \
SAV_RAW_DIR=./_data/raw/sav_extracted \
./download.sh
```

This tells `download.sh` to skip evaluation downloads and convert existing raw
SA-1B/SA-V extracts into the normal prepared roots.

If `SA1B_RAW_DIR` or `SAV_RAW_DIR` is not set, `download.sh` may download from
the manifests into:

```text
$RAW_DIR/sa1b_archives -> $RAW_DIR/sa1b_extracted -> $SA1B_DIR
$RAW_DIR/sav_archives  -> $RAW_DIR/sav_extracted  -> $SAV_DIR
```

Evaluation data is separate:

```text
VAL_ROOT_DAVIS or DATA_ROOT/DAVIS
VAL_ROOT_MOSE  or DATA_ROOT/MOSE
VAL_ROOT_LVOS  or DATA_ROOT/LVOS
VAL_ROOT_SAV   or DATA_ROOT/SA-V-test
VAL_ROOT_YTVOS or DATA_ROOT/YTVOS2019
```

`smoke_test.sh` and `train.sh` may resolve these parent directories to a usable
split subdirectory when needed, such as `MOSE/valid`, `LVOS/train`,
`SA-V-test/sav_test`, or `YTVOS2019/valid`.

Evaluation roots must never be used as substitutes for SA-1B or SA-V training
data.

## `smoke_test.sh`

`smoke_test.sh` is a consumer of prepared data. It should not consume raw
archives or raw extracted caches directly.

Image root resolution:

```text
DATA_ROOT_IMAGE
DATA_ROOT/sa1b
DATA_ROOT
```

Video root resolution:

```text
DATA_ROOT_VIDEO
DATA_ROOT/sav
DATA_ROOT/sav_sample  (legacy fallback only)
DATA_ROOT
```

For the current `.env`, smoke training should succeed only after these exist:

```text
./_data/sa1b/images
./_data/sa1b/masks
./_data/sav/JPEGImages
./_data/sav/Annotations
```

If the video root is missing, the correct next step is to run `download.sh` with
`SAV_RAW_DIR=./_data/raw/sav_extracted`; it is not correct to point
`DATA_ROOT_VIDEO` at `_data/SA-V-test`.

Smoke evaluation may use any configured VOS evaluation root, including:

```text
VAL_ROOT_SAV=./_data/SA-V-test/sav_test
```

## `train.sh`

`train.sh` is also a consumer of prepared data, but for full training. It uses
the same training root contract as `download.sh`:

```text
DATA_ROOT_IMAGE, or DATA_ROOT/sa1b
DATA_ROOT_VIDEO, or DATA_ROOT/sav
```

It fails early if either prepared root is missing:

```text
images/ and masks/ for image training
JPEGImages/ and Annotations/ for video training
```

It also requires at least one VOS evaluation root:

```text
VAL_ROOT_DAVIS
VAL_ROOT_MOSE
VAL_ROOT_LVOS
VAL_ROOT_SAV
VAL_ROOT_YTVOS
```

The evaluation roots may use DAVIS-style layouts or SA-V test layout, but they
are validation inputs only. They do not satisfy `DATA_ROOT_VIDEO`.

## Script Communication Rule

The scripts communicate through prepared directories and environment variables,
not through implicit raw-cache guesses:

```text
raw archives/extracts -> download.sh -> prepared roots -> smoke_test.sh/train.sh
```

Recommended handoff for this workspace:

```bash
DOWNLOAD_DAVIS=0 DOWNLOAD_EVAL=0 \
SA1B_RAW_DIR=./_data/raw/sa1b_extracted \
SAV_RAW_DIR=./_data/raw/sav_extracted \
./download.sh

VAL_ROOT_SAV=./_data/SA-V-test/sav_test ./smoke_test.sh
```

After `download.sh` has prepared `./_data/sav`, `train.sh` can use the same
`.env` values without additional training-data overrides.
