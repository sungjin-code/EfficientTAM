# Efficient Track Anything

EfficientTAM is a lightweight image and video segmentation model adapted from
SAM 2. This repository contains the EfficientTAM model code plus local scripts
for data preparation, smoke testing, training, validation, and headless
inference.

Paper: [Efficient Track Anything, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Xiong_Efficient_Track_Anything_ICCV_2025_paper.pdf)

If you are a coding agent, read [AGENT.md](AGENT.md) before editing this
repository.

## What Is Here

```text
efficient_track_anything/   Model, predictors, configs, CUDA extension
training/                   Image/video training entry points and loops
tools/                      Inference and validation utilities
data/                       Dataset download and preparation helpers
app.py, app_image.py         Gradio video and image demos
download.sh                  Download and prepare paper data
smoke_test.sh                Quick prepared-data training/evaluation check
train.sh                     Paper-style two-stage training pipeline
main.sh                      Download, smoke test, then train
```

## Install

Use Python 3.12. CUDA is recommended for training.

```bash
conda create -n efficienttam python=3.12
conda activate efficienttam
pip install -r requirements.txt
```

Optional pretrained demo checkpoints:

```bash
cd checkpoints
./download_checkpoints.sh
cd ..
```

## Configure

Copy the example env file and fill in local paths:

```bash
cp .env.example .env
```

The simplest paper-style layout uses one shared `DATA_ROOT`:

```dotenv
DATA_ROOT=/data/efficienttam
OUTPUT_DIR=runs
WANDB_MODE=disabled
```

With only `DATA_ROOT` set, the scripts expect or prepare:

```text
DATA_ROOT/sa1b
DATA_ROOT/sav
DATA_ROOT/DAVIS
DATA_ROOT/MOSE
DATA_ROOT/LVOS
DATA_ROOT/SA-V-test
DATA_ROOT/YTVOS2019
DATA_ROOT/raw
```

Per-root overrides such as `DATA_ROOT_IMAGE`, `DATA_ROOT_VIDEO`,
`VAL_ROOT_DAVIS`, `VAL_ROOT_MOSE`, `VAL_ROOT_LVOS`, `VAL_ROOT_SAV`,
`VAL_ROOT_YTVOS`, and `RAW_DATA_DIR` are also supported.

## Quick Workflow

Run the full local pipeline:

```bash
./main.sh
```

Run individual stages:

```bash
./download.sh
./smoke_test.sh
./train.sh
```

Skip full-pipeline stages with:

```bash
RUN_DOWNLOAD=0 RUN_SMOKE_TEST=0 ./main.sh
```

## Train

Stage 1 image pretraining:

```bash
python -m training.train_image --config training/train_image_s
```

Stage 2 video fine-tuning:

```bash
python -m training.train_video \
  --config training/train_video_s \
  --init-from runs/image_s/image_final.pt
```

Add Hydra overrides after the regular flags:

```bash
python -m training.train_image \
  --config training/train_image_s \
  train.batch_size=4 train.epochs=2
```

For multi-GPU paper-style training, use `train.sh`:

```bash
NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 ./train.sh
```

## Validate

After video training, run the VOS benchmark suite:

```bash
python -m tools.validate_vos_suite \
  --config configs/efficienttam/efficienttam_ti_512x512.yaml \
  --ckpt runs/video_ti/video_final.pt \
  --output-json runs/video_ti/eval/vos_suite.json \
  --mose /data/MOSE/valid \
  --davis /data/DAVIS \
  --lvos /data/LVOS/val \
  --sav /data/SA-V-test/sav_test \
  --ytvos /data/YTVOS2019/valid
```

The suite reports the Table 1 accuracy columns from the paper:

```text
MOSE, DAVIS, LVOS, SA-V: J&F
YTVOS: G
```

Smoke runs write evaluation results under:

```text
_runs/smoke/video_ti/eval/vos_suite.json
```

## Inference

Image inference:

```bash
python -m tools.infer --mode image \
  --config configs/efficienttam/efficienttam_s.yaml \
  --ckpt /path/to/checkpoint.pt \
  --input /path/to/image.jpg \
  --prompt /path/to/prompt.json \
  --output /path/to/mask.png
```

Video inference:

```bash
python -m tools.infer --mode video \
  --config configs/efficienttam/efficienttam_s.yaml \
  --ckpt /path/to/checkpoint.pt \
  --input /path/to/frame_dir \
  --prompt /path/to/prompt.json \
  --output /path/to/output_masks
```

Prompt examples:

```json
{"points": [[500, 300]], "labels": [1]}
{"box": [120, 80, 640, 520]}
{"mask": "/path/to/binary_mask.png"}
```

## Outputs

Training writes checkpoints and machine-readable artifacts:

```text
image_latest.pt
image_final.pt
image_training_artifact.json
video_latest.pt
video_final.pt
video_training_artifact.json
```

The JSON artifacts are useful for checking stage status, step counts, final
training losses, checkpoint paths, and checkpoint sizes.

## Demos

```bash
python app_image.py
python app.py
```

## Troubleshooting

- Missing package: activate the environment and reinstall `requirements.txt`.
- CUDA OOM: lower `train.batch_size`, `train.objects_per_image`,
  `train.objects_per_clip`, or `train.clip_len`.
- Missing data: check that image data has `images/` and `masks/`, and video data
  has `JPEGImages/` and `Annotations/`.
- Disable W&B: set `WANDB_MODE=disabled`.

## Acknowledgements

- [EfficientTAM](https://github.com/yformer/EfficientTAM)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [EfficientSAM](https://github.com/yformer/EfficientSAM)
- [SAM](https://github.com/facebookresearch/segment-anything)
