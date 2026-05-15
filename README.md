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
└── configs/                        # train_{image,video}_{s,ti}.yaml
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

Pick a variant when you start a training run by selecting the matching training config under `training/configs/`:

| Variant       | Resolution | Training configs                                   | Notes                      |
| ------------- | ---------- | -------------------------------------------------- | -------------------------- |
| `s`           | 1024×1024  | `train_image_s.yaml`, `train_video_s.yaml`         | Small model, best quality  |
| `s_512x512`   | 512×512    | (use the `_s_512x512` model YAML in overrides)     | Small model, faster        |
| `ti`          | 1024×1024  | `train_image_ti.yaml`, `train_video_ti.yaml`       | Tiny model, most efficient |
| `ti_512x512`  | 512×512    | (use the `_ti_512x512` model YAML in overrides)    | Tiny model, fastest        |

Architecture YAMLs live in `efficient_track_anything/configs/efficienttam/`; training YAMLs (which reference an architecture YAML plus hyperparameters) live in `training/configs/`.

---

## Training

EfficientTAM is trained in two stages, matching the paper recipe.

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
python -m training.train_image \
    --config training/configs/train_image_s.yaml \
    --data-root /path/to/sa1b_style_root \
    --output-dir runs/image_s
```

Key flags:

- `--resume <ckpt>` — resume the same stage (loads optimizer + scheduler too)
- `--overfit-one-batch` — single-batch smoke test; loss should drop near zero in a few hundred steps
- All hyperparameters live in [training/configs/train_image_s.yaml](training/configs/train_image_s.yaml) (or `train_image_ti.yaml` for the tiny variant)

### Stage 2: video fine-tuning

```bash
python -m training.train_video \
    --config training/configs/train_video_s.yaml \
    --data-root /path/to/sav_or_davis_root \
    --output-dir runs/video_s \
    --init-from runs/image_s/image_final.pt
```

`--init-from` seeds weights from the stage-1 checkpoint (memory_encoder / memory_attention keys load as random init since stage 1 doesn't train them).

Each video sample is an 8-frame clip with random stride; frame 0 gets a point or box prompt; a random subset of later frames receives a simulated correction click drawn from the model's own error region.

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
