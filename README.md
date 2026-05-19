# Efficient Track Anything (EfficientTAM)

> [Efficient Track Anything](https://arxiv.org/pdf/2411.18933), ICCV 2025.

A lightweight, real-time image and video segmentation model adapted from Meta's SAM2. EfficientTAM replaces SAM2's heavyweight ViT encoder with a compact ViT-Det backbone and introduces an efficient memory cross-attention mechanism, achieving >10 fps on iPhone 15 while maintaining segmentation quality comparable to SAM2.

---

## Repository Structure

```text
EfficientTAM/
├── app.py                          # Gradio demo — video segmentation (interactive)
├── app_image.py                    # Gradio demo — image segmentation (interactive)
├── checkpoints/                    # Destination for weights saved by your own training runs
├── notebooks/                      # Example scripts and Jupyter notebooks
├── examples/                       # Sample images and videos for testing
├── efficient_track_anything/       # Core model package (inference + architecture)
├── training/                       # From-scratch training pipeline
└── tools/                          # Batch inference + J&F validation
```

### `efficient_track_anything/` — model + inference

```text
efficient_track_anything/
├── build_efficienttam.py           # Model factory — entry point for loading models
├── efficienttam_video_predictor.py # Stateful video tracking interface
├── efficienttam_image_predictor.py # Stateless image segmentation interface
├── automatic_mask_generator.py     # Segment-everything via grid-based prompting
├── benchmark.py                    # FPS and parameter count benchmarking
├── configs/efficienttam/           # Hydra YAML configs for each model variant
├── modeling/                       # Core model architecture
│   ├── efficienttam_base.py        # EfficientTAMBase — shared model class
│   ├── memory_attention.py         # Efficient cross-attention between frames
│   ├── memory_encoder.py           # Encodes output masks into memory tokens
│   ├── position_encoding.py        # RoPE positional encoding
│   ├── backbones/vitdet.py         # Lightweight ViT-Det image encoder
│   └── sam/                        # Prompt encoder + mask decoder (from SAM2)
└── utils/                          # Image transforms, AMG helpers, video I/O
```

### `training/` — train EfficientTAM from scratch

```text
training/
├── train_image.py                  # Stage 1 entry: SA-1B-style image pretraining
├── train_video.py                  # Stage 2 entry: SA-V-style video fine-tuning
├── engine.py                       # forward_clip + train loop (manages memory bookkeeping)
├── losses.py                       # focal + dice + IoU L1 + obj-score BCE (best-of-3)
├── optim.py                        # AdamW + layerwise LR decay + warmup-cosine
├── data/
│   ├── image_dataset.py            # SA-1B-style image dataset
│   ├── video_dataset.py            # SA-V / DAVIS / YouTube-VOS layout video dataset
│   ├── augment.py                  # Image / clip augmentations (shared per-clip state)
│   └── prompts.py                  # PromptSampler — point / box / correction-click sampling
└── wandb_utils.py                  # Optional Weights & Biases logger (no-op without API key)
```

### `tools/` — batch inference + validation

```text
tools/
├── infer.py                        # Headless batch inference (image / video frames dir)
├── validate.py                     # DAVIS-style J&F validation runner
└── jf_metric.py                    # J (region) & F (boundary) metric implementation
```

---

## Model Variants

Pick a variant when you start a training run by selecting the matching Hydra training config name:

| Variant       | Resolution | Training configs                                                       | Notes                      |
| ------------- | ---------- | ---------------------------------------------------------------------- | -------------------------- |
| `s`           | 1024×1024  | `training/train_image_s`, `training/train_video_s`                     | Small model, best quality  |
| `s_512x512`   | 512×512    | (use the `_s_512x512` model YAML in overrides)                         | Small model, faster        |
| `ti`          | 1024×1024  | `training/train_image_ti`, `training/train_video_ti`                   | Tiny model, most efficient |
| `ti_512x512`  | 512×512    | (use the `_ti_512x512` model YAML in overrides)                        | Tiny model, fastest        |

All YAMLs live under `efficient_track_anything/configs/` (the package's Hydra config root): architecture YAMLs in `configs/efficienttam/`, training YAMLs (which reference an architecture YAML plus hyperparameters) in `configs/training/`. `--config` takes a Hydra config name relative to that root.

---

## Training

EfficientTAM is trained in two stages, matching the paper recipe.

### Configuration via `.env`

Paths to datasets, output directories, and seed weights can be set in a `.env`
file at the repo root so they don't have to be passed on the CLI every run.
Copy [.env.example](.env.example) → `.env` and fill in what you need. CLI
flags always override the env values.

```dotenv
# Weights & Biases — leave WANDB_API_KEY empty to disable logging entirely.
WANDB_API_KEY=...
WANDB_PROJECT=efficient-tam

# Dataset roots (shared, or per-stage with _IMAGE / _VIDEO suffix)
DATA_ROOT=/data
DATA_ROOT_IMAGE=/data/sa1b
DATA_ROOT_VIDEO=/data/davis17

# Output directories for checkpoints
OUTPUT_DIR_IMAGE=runs/image_s
OUTPUT_DIR_VIDEO=runs/video_s

# Stage-1 → Stage-2 weight handoff, and resume points
INIT_FROM=runs/image_s/image_final.pt
RESUME_VIDEO=runs/video_s/video_latest.pt
```

Resolution order for each path: `--cli-flag` → `${VAR}_IMAGE` / `${VAR}_VIDEO`
(stage-specific) → `${VAR}` (shared) → error if still unset.

### Dataset layout

**Stage 1 — image (SA-1B-style):**

```text
{data_root}/
├── images/
│   ├── img001.jpg
│   └── ...
└── masks/
    ├── img001.png            # binary PNG, one mask per file
    └── img001_2.png          # multi-object: append _{obj_id}
```

**Stage 2 — video (DAVIS / YouTube-VOS / SA-V layout):**

```text
{data_root}/
├── JPEGImages/
│   └── {video_id}/
│       ├── 00000.jpg
│       └── ...
└── Annotations/
    └── {video_id}/
        ├── 00000.png         # palette PNG; pixel value = object id (0=bg)
        └── ...
```

### Stage 1: image pretraining

```bash
# All paths can come from `.env`:
python -m training.train_image --config training/train_image_s

# Or pass them explicitly (overrides `.env`):
python -m training.train_image \
    --config training/train_image_s \
    --data-root /path/to/sa1b_style_root \
    --output-dir runs/image_s \
    train.batch_size=4 train.epochs=10   # optional Hydra-style overrides
```

Key flags:

- `--config` — Hydra config name under `efficient_track_anything/configs/` (e.g. `training/train_image_s`).
- `--resume <ckpt>` — resume the same stage (loads optimizer + scheduler too). Defaults to `$RESUME_IMAGE` / `$RESUME`.
- `--overfit-one-batch` — single-batch smoke test; loss should drop near zero in a few hundred steps.
- Positional `key=value` args at the end are Hydra overrides for the training YAML (`train.batch_size=8`, `train.epochs=2`, …).
- All hyperparameters live in [efficient_track_anything/configs/training/train_image_s.yaml](efficient_track_anything/configs/training/train_image_s.yaml) (or `train_image_ti.yaml` for the tiny variant).

### Stage 2: video fine-tuning

```bash
python -m training.train_video \
    --config training/train_video_s \
    --data-root /path/to/sav_or_davis_root \
    --output-dir runs/video_s \
    --init-from runs/image_s/image_final.pt
```

`--init-from` seeds weights from the stage-1 checkpoint (memory_encoder / memory_attention keys load as random init since stage 1 doesn't train them). It defaults to `$INIT_FROM_VIDEO` / `$INIT_FROM` from `.env`.

Each video sample is an 8-frame clip with random stride; frame 0 gets a point or box prompt; a random subset of later frames receives a simulated correction click drawn from the model's own error region.

### Checkpoints and crash safety

Each epoch writes both `*_epoch_NNNN.pt` and a `*_latest.pt` pointer for easy resume. The writes are atomic (`tmp` + rename) so a kill / OOM mid-write cannot corrupt a checkpoint.

On `KeyboardInterrupt` or any unhandled exception the trainer:

1. prints the traceback to stderr,
2. saves an emergency checkpoint to `*_interrupt.pt` and refreshes `*_latest.pt`,
3. flushes the wandb run with `result/status = interrupted | crashed`,
4. exits 1 only on crash (not on Ctrl-C).

A non-finite loss (NaN / Inf) skips the optimizer step instead of poisoning every parameter, logs a warning to stdout, and records `train/nan_skip=1` to wandb.

### Experiment tracking (Weights & Biases)

Training auto-logs to W&B when `WANDB_API_KEY` is set in `.env` (or the process env). Without a key it prints one notice and runs normally — wandb is completely optional. What's logged:

- **Config** — the full Hydra-composed YAML (`model.*`, `train.*`), CLI overrides, device, precision, and resolved path args.
- **Process** — per step: `train/loss`, `train/lr`, `train/epoch`, plus each loss component (`train/focal`, `train/dice`, `train/iou_l1`, `train/obj_bce`). Per `log_every` steps: rolling `*_avg` values and throughput (`train/ips` for image, `train/clips_per_s` for video). Per epoch: `checkpoint/epoch` marker.
- **Results** — written to the run summary: `result/status`, `result/final_checkpoint`, `result/steps`, `result/epochs`, and the last-epoch final loss + components.

Add an optional `wandb:` block to a training YAML to customize:

```yaml
wandb:
  project: efficient-tam
  run_name: image_s_run01
  tags: [stage1, paper]
```

### Training recipe (matches paper / SAM2)

- **Losses:** focal (weight 20) + dice (1) on masks, L1 on IoU prediction (1), BCE on object-score (1).
- **Best-of-3:** for multi-mask outputs, supervise the candidate with the lowest combined focal+dice loss.
- **Optimizer:** AdamW with weight decay 0.1 (bias/LN excluded), layer-wise LR decay 0.8 on the image encoder.
- **Schedule:** linear warmup (5% of steps) → cosine to 0.
- **Precision:** bf16 autocast on CUDA, fp16 on MPS, fp32 on CPU.
- **Gradient clipping:** 0.1 (SAM2 default).

---

## Evaluation (DAVIS-style J&F)

```bash
python -m tools.validate \
    --config configs/efficienttam/efficienttam_s.yaml \
    --ckpt runs/video_s/video_final.pt \
    --val-root /path/to/DAVIS17/val \
    --output-json results.json
```

This prompts every annotated object in frame 0 with its GT mask, propagates through the clip, and reports per-video J / F / J&F plus the overall means. Run this after Stage 2 finishes to score the checkpoint you just trained.

Flags:

- `--max-videos N` — evaluate a subset for quick iteration
- `--output-json results.json` — dump per-video metrics

The metric implementation is vendored from DAVIS 2017 (region IoU + boundary F-measure with disk-shaped dilation tolerance) and lives in [tools/jf_metric.py](tools/jf_metric.py).

---

## Inference

### Programmatic — single image

```python
import numpy as np
from PIL import Image
from efficient_track_anything.build_efficienttam import build_efficienttam
from efficient_track_anything.efficienttam_image_predictor import EfficientTAMImagePredictor

model = build_efficienttam("configs/efficienttam/efficienttam_s.yaml", "runs/video_s/video_final.pt")
predictor = EfficientTAMImagePredictor(model)
predictor.set_image(np.array(Image.open("examples/sf.jpg").convert("RGB")))

masks, iou, _ = predictor.predict(
    point_coords=np.array([[500, 300]]),
    point_labels=np.array([1]),
    multimask_output=True,
)
best = int(np.argmax(iou))
Image.fromarray((masks[best] * 255).astype("uint8")).save("mask.png")
```

### Programmatic — video tracking

```python
import torch
from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor

predictor = build_efficienttam_video_predictor(
    "configs/efficienttam/efficienttam_s.yaml",
    "runs/video_s/video_final.pt",
)
state = predictor.init_state("/path/to/frames/")          # dir of JPEGs

predictor.add_new_points_or_box(
    state, frame_idx=0, obj_id=1,
    points=[[x, y]], labels=[1],
)

for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
    masks = (mask_logits > 0).cpu().numpy()
    # ... save / visualize ...
```

### Headless batch inference (CLI)

```bash
# image
python -m tools.infer --mode image \
    --config configs/efficienttam/efficienttam_s.yaml \
    --ckpt runs/video_s/video_final.pt \
    --input examples/sf.jpg \
    --prompt prompt.json \
    --output sf_mask.png

# video (directory of JPEG frames)
python -m tools.infer --mode video \
    --config configs/efficienttam/efficienttam_s.yaml \
    --ckpt runs/video_s/video_final.pt \
    --input /path/to/frames/ \
    --prompt prompt.json \
    --output out_masks/
```

**Prompt JSON — image mode** (any one of):

```json
{"points": [[x, y]], "labels": [1]}
{"box": [x0, y0, x1, y1]}
{"mask": "/path/to/binary.png"}
```

**Prompt JSON — video mode:**

```json
{"objects": [
    {"obj_id": 1, "frame_idx": 0, "points": [[x, y]], "labels": [1]},
    {"obj_id": 2, "frame_idx": 0, "box":    [x0, y0, x1, y1]},
    {"obj_id": 3, "frame_idx": 0, "mask":   "/path/to/m.png"}
]}
```

### Interactive demo (Gradio)

The two `app*.py` scripts launch a Gradio UI for clicking/drawing prompts directly on images or videos:

```bash
python app_image.py          # image: point / box / segment-everything modes
python app.py                # video: frame-by-frame tracking with click prompts
```

The video demo writes annotated frames + an MP4 to the working directory; the image demo overlays masks live in the browser. Both auto-pick CUDA / MPS / CPU.

### Benchmark FPS

```bash
python efficient_track_anything/benchmark.py
```

---

## Acknowledgements

- [EfficientTAM](https://github.com/yformer/EfficientTAM)
- [SAM2](https://github.com/facebookresearch/sam2)
- [EfficientSAM](https://github.com/yformer/EfficientSAM)
- [SAM](https://github.com/facebookresearch/segment-anything)
