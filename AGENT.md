# Agent Guide

If you are a coding agent working in this repository, read this file before editing code or scripts. It explains the project shape, data policy, common commands, and the artifacts that should be checked after training or smoke tests.

Use `paper.txt` as the local reference for the EfficientTAM paper when checking training recipe details, evaluation datasets, metric names, and Table 1 semantics.

## Project Shape

EfficientTAM is a Python/PyTorch image and video segmentation project.

```text
efficient_track_anything/   Core model package and predictors
training/                   Training entry points, engine, losses, data loaders
tools/                      Batch inference and validation utilities
data/                       Dataset preparation helpers and manifests
_scripts/                   Split training/evaluation shell helpers
app.py                      Interactive Gradio video demo
app_image.py                Interactive Gradio image demo
```

Hydra configs are packaged under `efficient_track_anything/configs/`:

- `efficienttam/`: model architecture configs.
- `training/`: stage-specific training configs.

Training entry points accept config names such as `training/train_image_ti` or `training/train_video_s`. They also tolerate legacy-looking paths such as `training/configs/train_image_s.yaml`, but those are normalized to `efficient_track_anything/configs/training/*.yaml`; there is no `training/configs/` directory in the repository.

The codebase is intentionally compact. Prefer existing patterns and local helper APIs over new abstractions.

## Environment

Use Python 3.12 and install dependencies from `requirements.txt`.

```bash
pip install -r requirements.txt
```

The dependency set includes PyTorch, Hydra, Gradio, OpenCV, SciPy, pytest, Hugging Face Hub, `gdown`, and optional W&B support.

Local paths and W&B settings are loaded from `.env`. Do not commit local dataset paths, run directories, API keys, or checkpoints.

## Path Resolution

Training path resolution order is:

1. CLI flag.
2. Stage-specific env var, such as `DATA_ROOT_IMAGE` or `OUTPUT_DIR_VIDEO`.
3. Shared env var, such as `DATA_ROOT` or `OUTPUT_DIR`.
4. Hard error if still unset.

With only `DATA_ROOT` set, the paper pipeline expects:

```text
{DATA_ROOT}/sa1b
{DATA_ROOT}/sav
{DATA_ROOT}/DAVIS
{DATA_ROOT}/MOSE
{DATA_ROOT}/LVOS
{DATA_ROOT}/SA-V-test
{DATA_ROOT}/YTVOS2019
{DATA_ROOT}/raw
```

Per-root overrides take precedence:

```text
DATA_ROOT_IMAGE
DATA_ROOT_VIDEO
VAL_ROOT_DAVIS
VAL_ROOT_MOSE
VAL_ROOT_LVOS
VAL_ROOT_SAV
VAL_ROOT_YTVOS
RAW_DATA_DIR
OUTPUT_DIR_IMAGE
OUTPUT_DIR_VIDEO
```

Detailed data-path behavior lives in `data/README.md`. Treat that file as the source of truth for how `download.sh`, `smoke_test.sh`, and `train.sh` pass data between archive caches, extracted caches, prepared training roots, and validation roots.

`_scripts/train_common.sh` contains the newer shared shell resolution logic used by split stage scripts. Keep root resolution, variant defaults, auto-batch, and evaluation behavior in sync with `train.sh`/`smoke_test.sh` when changing it.

## Data Policy

Training data must be paper data only:

- Stage 1 uses SA-1B-style image data.
- Stage 2 uses SA-V-style video data plus the configured SA-1B image mix.
- DAVIS, MOSE, LVOS, SA-V test, and YTVOS are validation/evaluation roots, not substitutes for SA-1B or SA-V training.

Do not reintroduce a DAVIS-derived training fallback. If paper data is missing, fail with a clear error instead of silently training on validation data.

Dataset manifests:

```text
data/SA-1B.txt
data/SA-V.txt
```

`download.sh` reads these manifests, downloads archives into `{DATA_ROOT}/raw`, extracts them, and converts them to prepared training layouts. Google Drive evaluation downloads require `gdown`; Hugging Face evaluation downloads require `huggingface-cli`.

When changing data handling, keep `data/README.md` in sync. Preserve the distinction between:

- raw archive/cache roots under `RAW_DATA_DIR`;
- prepared training roots such as `DATA_ROOT/sa1b` and `DATA_ROOT/sav`;
- validation roots such as `DATA_ROOT/SA-V-test`.

## Data Layouts

Image training expects:

```text
{data_root}/images/*.jpg
{data_root}/masks/*.png
```

Video training expects DAVIS / YouTube-VOS style directories:

```text
{data_root}/JPEGImages/{video_id}/*.jpg
{data_root}/Annotations/{video_id}/*.png
```

VOS validation accepts:

- `JPEGImages/Annotations`
- DAVIS `JPEGImages/480p` and `Annotations/480p`
- SA-V `JPEGImages_24fps/Annotations_6fps`
- the same layouts under `valid/`, `val/`, or `test/`

Some validation datasets have sparse annotations. The evaluator should propagate through all frames but score only frames that have annotation masks.

SA-23 image validation is optional and uses `SA23_ROOT` or `tools.validate_image_miou --root`. It expects the same image layout as Stage 1:

```text
{SA23_ROOT}/images/*.jpg
{SA23_ROOT}/masks/*.png
```

## Common Commands

Complete local workflow:

```bash
./main.sh
```

`main.sh` runs dataset download/preparation, smoke training, then full two-stage paper training and validation. Skip stages with:

```bash
RUN_DOWNLOAD=0
RUN_SMOKE_TEST=0
RUN_TRAIN=0
```

Download and prepare paper datasets:

```bash
./download.sh
```

Run the prepared-data smoke test:

```bash
./smoke_test.sh
```

Run paper training and validation after data is ready:

```bash
./train.sh
```

Run the newer split pipeline with resume/eval helpers:

```bash
_scripts/train_pipeline.sh
```

Run split stages after an interruption or when evaluating an existing video checkpoint:

```bash
_scripts/train_sa1b_stage1.sh
_scripts/train_sav_stage2.sh
_scripts/train_eval.sh
```

Important script knobs:

```text
VARIANT=ti|s
NPROC_PER_NODE
CUDA_VISIBLE_DEVICES
TARGET_GLOBAL_BATCH
AUTO_BATCH
IMAGE_BATCH_PER_GPU
VIDEO_BATCH_PER_GPU
EVAL_GPUS
SA23_ROOT
```

Stage 1 image pretraining:

```bash
python -m training.train_image --config training/train_image_s
```

Stage 2 video fine-tuning:

```bash
python -m training.train_video --config training/train_video_s
```

Hydra overrides go after regular flags:

```bash
python -m training.train_image --config training/train_image_s train.batch_size=4 train.epochs=2
```

The entry points also accept:

```text
--data-root
--image-data-root      # Stage 2 image mix only
--output-dir
--resume
--init-from
--overfit-one-batch
--max-steps
--precision auto|bf16|fp16|fp32
```

Headless image inference:

```bash
python -m tools.infer --mode image \
  --config configs/efficienttam/efficienttam_s.yaml \
  --ckpt /path/to/checkpoint.pt \
  --input /path/to/image.jpg \
  --prompt /path/to/prompt.json \
  --output /path/to/mask.png
```

Run tests if tests are added:

```bash
python -m pytest
```

There is currently no `tests/` directory or packaging metadata such as `pyproject.toml`.

## Validation Metrics

`tools.validate_vos_suite` is the paper-style VOS benchmark runner. It writes JSON to the path passed via `--output-json`.

The paper Table 1 accuracy metrics are:

```text
MOSE val:       J&F
DAVIS 2017 val: J&F
LVOS val:       J&F
SA-V test:      J&F
YTVOS 2019 val: G
```

The suite output should preserve those names:

```text
benchmarks.<name>.primary_metric
benchmarks.<name>.primary_score
mean_primary_score_over_evaluated_benchmarks
```

For YTVOS, use `G` / `G_mean` in suite output, not `JF` / `JF_mean`. The value is computed from J and F but the reported paper metric name is `G`.

Smoke-test scores are not expected to match the paper. They only verify that the pipeline runs and that metric columns are produced.

The VOS suite supports resumable per-benchmark output and strided sharding:

```bash
python -m tools.validate_vos_suite \
  --config configs/efficienttam/efficienttam_ti.yaml \
  --ckpt runs/video_ti/video_final.pt \
  --output-json runs/video_ti/eval/vos_suite.shard0.json \
  --num-shards 2 --shard-idx 0

python -m tools.merge_vos_shards \
  --output-json runs/video_ti/eval/vos_suite.json \
  runs/video_ti/eval/vos_suite.shard0.json \
  runs/video_ti/eval/vos_suite.shard1.json
```

`_scripts/train_eval.sh` does this automatically when `EVAL_GPUS` resolves to more than one GPU. It also sets `VAL_ANN_ROOT_MOSE` by default for MOSEv2-style layouts where dense masks live outside `valid/Annotations`.

SA-23-style image mIoU is separate from Table 1 VOS metrics:

```bash
python -m tools.validate_image_miou \
  --config configs/efficienttam/efficienttam_ti.yaml \
  --ckpt runs/video_ti/video_final.pt \
  --root "$SA23_ROOT" \
  --output-json runs/video_ti/eval/sa23_miou.json
```

## Training Artifacts

Rank 0 writes checkpoints and machine-readable artifacts for each stage:

```text
image_epoch_0000.pt
image_latest.pt
image_final.pt
image_training_artifact.json
video_epoch_0000.pt
video_latest.pt
video_final.pt
video_training_artifact.json
```

The JSON artifacts include:

- stage status;
- step and epoch;
- target steps;
- final training metrics;
- checkpoint paths;
- checkpoint existence;
- checkpoint sizes.

Use these artifacts to verify that training completed and that final checkpoints are usable. Evaluation metrics are written separately, usually under `<video_output>/eval/`.

## Smoke Test Outputs

Smoke defaults write under `runs/smoke` unless `OUTPUT_DIR` or `SMOKE_OUTPUT_ROOT` points elsewhere:

```text
runs/smoke/image_ti/image_training_artifact.json
runs/smoke/video_ti/video_training_artifact.json
runs/smoke/video_ti/eval/vos_suite.json
runs/smoke/video_ti/eval/sa23_miou.json
```

The training artifacts contain training losses and checkpoint metadata. VOS metrics live in `vos_suite.json`; optional SA-23 image metrics live in `sa23_miou.json`.

## Development Notes

- Use `rg` or `rg --files` for searches.
- Keep edits scoped to the affected training, model, data, or tool module.
- Preserve `.env`, checkpoints, and local run outputs as local-only state.
- Do not revert user changes in a dirty worktree.
- Prefer structured parsers/APIs over ad hoc string manipulation.
- The CUDA extension source is `efficient_track_anything/csrc/connected_components.cu`.
- Check GPU availability before assuming CUDA-only behavior is exercised.
- Use `bash -n` for changed shell scripts.
- Use `python -m py_compile` for changed Python entry points when practical.
- Prefer `tools.validate_vos_suite` for paper-style VOS reporting; use `tools.validate` only for single-root DAVIS-style debugging.

## Documentation Rules

Keep README short and human-facing. Put implementation details, data policy, agent notes, and validation semantics in this file or in `data/README.md`.

Do not hard-wrap normal prose in Markdown. Keep each paragraph, bullet item, and blockquote sentence on a clean logical line unless a line break is required for Markdown syntax, a table, or a code block.

All documentation and code comments should be written in English.
