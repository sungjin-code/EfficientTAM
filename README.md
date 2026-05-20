# Efficient Track Anything (EfficientTAM)

> [Efficient Track Anything](https://openaccess.thecvf.com/content/ICCV2025/papers/Xiong_Efficient_Track_Anything_ICCV_2025_paper.pdf), ICCV 2025.

This repository contains the official EfficientTAM model/inference code plus a
local training, validation, and headless inference pipeline.

EfficientTAM is a lightweight image/video segmentation model adapted from SAM 2.
It uses compact ViT backbones and an efficient memory attention module to reduce
latency while keeping SAM 2-like segmentation quality.

## Contents

```text
EfficientTAM/
├── app.py                          # Gradio video demo
├── app_image.py                    # Gradio image demo
├── checkpoints/                    # Downloaded or trained weights
├── data/                           # Dataset preparation helpers
├── efficient_track_anything/       # Official EfficientTAM package
│   ├── build_efficienttam.py       # Model / predictor builders
│   ├── configs/efficienttam/       # Model architecture configs
│   ├── configs/training/           # Training configs
│   ├── efficienttam_image_predictor.py
│   ├── efficienttam_video_predictor.py
│   └── modeling/
├── training/                       # Local training pipeline
│   ├── train_image.py              # Stage 1 image pretraining
│   ├── train_video.py              # Stage 2 video fine-tuning
│   ├── distributed.py              # torchrun helpers for multi-GPU
│   ├── engine.py                   # forward / train loop / checkpoints
│   ├── losses.py                   # focal + dice + IoU + object losses
│   ├── optim.py                    # AdamW + paper LR schedules
│   └── data/
├── tools/
│   ├── infer.py                    # CLI image/video inference
│   ├── validate.py                 # DAVIS-style J&F validation
│   └── jf_metric.py
└── train.sh                        # 2-GPU training pipeline
```

## Install

Use Linux with CUDA for training. The training scripts auto-select CUDA, MPS, or
CPU, but the intended setup is CUDA.

```bash
conda create -n efficienttam python=3.12
conda activate efficienttam
pip install -r requirements.txt
```

If you want to run the official interactive demos with pretrained weights:

```bash
cd checkpoints
./download_checkpoints.sh
cd ..
```

## Configure Paths

Copy the example env file and fill in your local paths:

```bash
cp .env.example .env
```

Typical `.env`:

```dotenv
WANDB_API_KEY=
WANDB_PROJECT=efficient-tam

DATA_ROOT_IMAGE=/data/sa1b_style
DATA_ROOT_VIDEO=/data/davis_or_sav

OUTPUT_DIR_IMAGE=runs/image_ti
OUTPUT_DIR_VIDEO=runs/video_ti

INIT_FROM=runs/image_ti/image_final.pt
```

Path resolution order is:

```text
CLI flag -> stage-specific env var -> shared env var -> error
```

For example, `training.train_image` resolves data root as
`--data-root`, then `DATA_ROOT_IMAGE`, then `DATA_ROOT`.

## Dataset Layout

### Stage 1: Image Pretraining

The image dataset uses an SA-1B-style folder layout:

```text
{DATA_ROOT_IMAGE}/
├── images/
│   ├── img001.jpg
│   └── img002.png
└── masks/
    ├── img001.png
    ├── img001_2.png
    └── img002.png
```

Mask files are binary PNGs. Multiple masks for one image can be stored as
`{image_id}_{object_id}.png`.

### Stage 2: Video Fine-Tuning / Validation

The video dataset uses DAVIS / YouTube-VOS / SA-V style directories:

```text
{DATA_ROOT_VIDEO}/
├── JPEGImages/
│   └── {video_id}/
│       ├── 00000.jpg
│       └── 00001.jpg
└── Annotations/
    └── {video_id}/
        ├── 00000.png
        └── 00001.png
```

Annotation PNGs are palette or grayscale masks where `0` is background and each
non-zero pixel value is an object id.

## Model / Training Configs

Training configs are Hydra config names under
`efficient_track_anything/configs/training/`.

| Variant | Image config | Video config | Resolution | Notes |
| --- | --- | --- | --- | --- |
| Tiny | `training/train_image_ti` | `training/train_video_ti` | 1024 | Lower VRAM |
| Small | `training/train_image_s` | `training/train_video_s` | 1024 | Better quality |
| Smoke | `training/_image_test` | `training/_video_test` | 512 | Quick sanity test |

Architecture configs live in `efficient_track_anything/configs/efficienttam/`.
For example, `training/train_image_ti` points at
`configs/efficienttam/efficienttam_ti.yaml`.

## Paper Recipe Notes

The local configs follow the EfficientTAM paper as closely as this pipeline
supports:

- Image encoder initialization: `training/train_image_{ti,s}` starts from the
  SAMI-pretrained EfficientSAM ViT-Ti / ViT-S image encoder by default. The
  weights are downloaded through PyTorch's model cache on first use.
- Stage 1 image pretraining: SA-1B-style images, 90k optimizer steps, LR `4e-4`,
  inverse-square-root decay, 1k warmup, 5k cooldown, bf16 on CUDA, focal:dice
  loss `20:1`.
- Stage 2 video fine-tuning: 300k optimizer steps, encoder LR `6e-5`, other
  module LR `3e-4`, cosine schedule, 15k warmup, loss `20:1:1:1`, with a 10%
  SA-1B image mix by default.
- Optimizer: AdamW, betas `(0.9, 0.999)`, weight decay `0.1`, layer-wise decay
  `0.8`, gradient clipping `0.1`.

Important practical differences:

- The paper uses global batch 256 on 256 A100-80G GPUs. On 2 GPUs, reduce
  `train.batch_size`, `train.objects_per_image`, or `train.objects_per_clip` if
  you hit OOM.
- On small GPU counts, `train.sh` uses gradient accumulation to keep the
  effective global batch at 256.

## Train

### Two-GPU Training on GPU 0 and 1

Use `torchrun` with `CUDA_VISIBLE_DEVICES=0,1`.

Stage 1:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --module training.train_image \
  --config training/train_image_ti
```

Stage 2:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --module training.train_video \
  --config training/train_video_ti \
  --init-from runs/image_ti/image_final.pt
```

The scripts also read `.env`, so the path flags can be omitted when `.env` is
filled in.

### Explicit Paths

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --module training.train_image \
  --config training/train_image_s \
  --data-root /data/sa1b_style \
  --output-dir runs/image_s
```

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --module training.train_video \
  --config training/train_video_s \
  --data-root /data/sav_or_davis \
  --output-dir runs/video_s \
  --init-from runs/image_s/image_final.pt
```

### Memory-Friendly Overrides

Hydra-style overrides go at the end:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --module training.train_image \
  --config training/train_image_ti \
  train.batch_size=1 train.objects_per_image=8 train.num_workers=2
```

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --module training.train_video \
  --config training/train_video_ti \
  train.batch_size=1 train.objects_per_clip=1 train.clip_len=4
```

Useful flags:

- `--resume <path>`: resume optimizer, scheduler, epoch, and step.
- `--init-from <path>`: load model weights only.
- `--overfit-one-batch`: single-batch debugging mode.
- `--max-steps N`: override the config step budget.
- `--precision auto|bf16|fp16|fp32`: mixed precision mode.

### Paper Training Pipeline

`train.sh` runs the paper-aligned 1024px two-stage training pipeline on GPU 0 and
1, using gradient accumulation so the effective global batch is 256. After
training, it runs the configured VOS benchmark roots and optional SA-23 mIoU:

```bash
NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 ./train.sh
```

Set `DATA_ROOT_IMAGE`, `DATA_ROOT_VIDEO`, at least one `VAL_ROOT_*` VOS root, and
optionally `SA23_ROOT` / `OUTPUT_DIR` in `.env` before using the script. Use
`VARIANT=s ./train.sh` for EfficientTAM-S; the default is `VARIANT=ti`. The
script verifies the training dataset layout before starting.

## Checkpoints

Training writes:

```text
image_epoch_0000.pt
image_latest.pt
image_final.pt
video_epoch_0000.pt
video_latest.pt
video_final.pt
```

Writes are atomic via `*.tmp` then rename. On interruption or crash, the trainer
tries to write `image_interrupt.pt` or `video_interrupt.pt` and refresh
`*_latest.pt`.

In multi-GPU training, rank 0 writes checkpoints and W&B logs.

## Validate

Run DAVIS-style J&F validation after Stage 2:

```bash
python -m tools.validate \
  --config configs/efficienttam/efficienttam_ti.yaml \
  --ckpt runs/video_ti/video_final.pt \
  --val-root /data/davis_or_sav \
  --output-json runs/video_ti/results.json
```

Quick subset:

```bash
python -m tools.validate \
  --config configs/efficienttam/efficienttam_ti.yaml \
  --ckpt runs/video_ti/video_final.pt \
  --val-root /data/davis_or_sav \
  --max-videos 5
```

Validation prompts every object in frame 0 with the ground-truth mask, propagates
through the sequence, and reports per-video `J`, `F`, and `J&F` plus overall
means.

## Inference

### CLI: Image

Create a prompt JSON:

```json
{"points": [[500, 300]], "labels": [1]}
```

Run inference:

```bash
python -m tools.infer --mode image \
  --config configs/efficienttam/efficienttam_ti.yaml \
  --ckpt runs/video_ti/video_final.pt \
  --input /path/to/image.jpg \
  --prompt prompt_image.json \
  --output out_mask.png
```

Image prompts can be point, box, or mask:

```json
{"points": [[500, 300]], "labels": [1]}
{"box": [120, 80, 640, 520]}
{"mask": "/path/to/binary_mask.png"}
```

### CLI: Video

Input is a directory of JPEG frames:

```text
/path/to/frames/
├── 00000.jpg
├── 00001.jpg
└── ...
```

Prompt JSON:

```json
{
  "objects": [
    {"obj_id": 1, "frame_idx": 0, "points": [[500, 300]], "labels": [1]},
    {"obj_id": 2, "frame_idx": 0, "box": [120, 80, 640, 520]}
  ]
}
```

Run:

```bash
python -m tools.infer --mode video \
  --config configs/efficienttam/efficienttam_ti.yaml \
  --ckpt runs/video_ti/video_final.pt \
  --input /path/to/frames \
  --prompt prompt_video.json \
  --output out_masks
```

Outputs are binary PNG masks named like:

```text
out_masks/obj01_frame00000.png
out_masks/obj01_frame00001.png
```

### Python: Image

```python
import numpy as np
from PIL import Image

from efficient_track_anything.build_efficienttam import build_efficienttam
from efficient_track_anything.efficienttam_image_predictor import EfficientTAMImagePredictor

model = build_efficienttam(
    "configs/efficienttam/efficienttam_ti.yaml",
    ckpt_path="runs/video_ti/video_final.pt",
)
predictor = EfficientTAMImagePredictor(model)
predictor.set_image(np.array(Image.open("/path/to/image.jpg").convert("RGB")))

masks, iou, _ = predictor.predict(
    point_coords=np.array([[500, 300]], dtype=np.float32),
    point_labels=np.array([1], dtype=np.int32),
    multimask_output=True,
)
best = int(np.argmax(iou))
Image.fromarray((masks[best] * 255).astype("uint8")).save("mask.png")
```

### Python: Video

```python
from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor

predictor = build_efficienttam_video_predictor(
    "configs/efficienttam/efficienttam_ti.yaml",
    "runs/video_ti/video_final.pt",
)
state = predictor.init_state("/path/to/frames")

predictor.add_new_points_or_box(
    state,
    frame_idx=0,
    obj_id=1,
    points=[[500, 300]],
    labels=[1],
)

for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
    masks = (mask_logits > 0).cpu().numpy()
```

## Interactive Demos

```bash
python app_image.py
python app.py
```

`app_image.py` launches the image demo. `app.py` launches the video tracking
demo. Both auto-pick CUDA, MPS, or CPU.

## Benchmark

```bash
python efficient_track_anything/benchmark.py
```

## W&B Logging

W&B is optional. If `WANDB_API_KEY` is empty, training runs normally without
logging.

You can customize runs in a training YAML:

```yaml
wandb:
  project: efficient-tam
  run_name: video_ti_run01
  tags: [stage2, ti]
```

## Troubleshooting

- `ModuleNotFoundError: torch`: activate the conda environment and install
  `requirements.txt`.
- CUDA OOM: lower `train.batch_size`, `train.objects_per_image`,
  `train.objects_per_clip`, or `train.clip_len`.
- No dataset found: check that `images/masks` or `JPEGImages/Annotations` exist
  under the selected data root.
- Resume training: pass `--resume runs/.../*_latest.pt`.
- Disable W&B even with a key: set `WANDB_MODE=disabled`.

## Acknowledgements

- [EfficientTAM](https://github.com/yformer/EfficientTAM)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [EfficientSAM](https://github.com/yformer/EfficientSAM)
- [SAM](https://github.com/facebookresearch/segment-anything)
