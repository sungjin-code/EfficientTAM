"""Video segmentation dataset for Stage 2 (SA-V / DAVIS / YouTube-VOS-style layout).

Expected layout:

    {root}/
        JPEGImages/
            {video_id}/
                00000.jpg
                00001.jpg
                ...
        Annotations/
            {video_id}/
                00000.png   # palette PNG; pixel values = object ids (0 = background)
                ...

Per sample we:
  - pick a video,
  - pick an object id present in the first sampled frame,
  - sample `clip_len` consecutive frames at a random stride,
  - return frames + per-frame binary masks for the chosen object,
  - generate a frame-0 prompt and decide which later frames receive correction clicks.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from training.data.augment import apply_image, normalize, sample_params
from training.data.prompts import PromptSample, PromptSampler


def _list_videos(root: Path) -> list[str]:
    jpeg = root / "JPEGImages"
    ann = root / "Annotations"
    if not jpeg.exists() or not ann.exists():
        raise FileNotFoundError(f"Expected {jpeg} and {ann} to exist")
    videos = sorted(
        [d.name for d in jpeg.iterdir() if d.is_dir() and (ann / d.name).is_dir()]
    )
    if not videos:
        raise RuntimeError(f"No videos found under {jpeg}")
    return videos


class VideoSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | os.PathLike,
        image_size: int,
        clip_len: int,
        prompt_sampler: PromptSampler,
        correction_prob: float = 0.5,
        max_correction_frames: int = 2,
        stride_choices: tuple[int, ...] = (1, 2, 3),
        scale_range: tuple[float, float] = (0.7, 1.0),
        hflip_prob: float = 0.5,
        brightness: float = 0.1,
        contrast: float = 0.03,
        saturation: float = 0.03,
        grayscale_prob: float = 0.05,
        affine_degree: float = 25.0,
        affine_shear: float = 20.0,
        min_mask_pixels: int = 32,
        objects_per_clip: int = 1,
        seed: Optional[int] = None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.clip_len = clip_len
        self.prompt_sampler = prompt_sampler
        self.correction_prob = correction_prob
        self.max_correction_frames = max_correction_frames
        self.stride_choices = stride_choices
        self.scale_range = scale_range
        self.hflip_prob = hflip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.grayscale_prob = grayscale_prob
        self.affine_degree = affine_degree
        self.affine_shear = affine_shear
        self.min_mask_pixels = min_mask_pixels
        self.objects_per_clip = objects_per_clip
        self.rng = random.Random(seed)
        self.videos = _list_videos(self.root)

    def __len__(self) -> int:
        return len(self.videos)

    def _list_frames(self, video: str) -> list[Path]:
        return sorted((self.root / "JPEGImages" / video).glob("*.jpg"))

    def _load_mask(self, video: str, frame_path: Path, obj_id: int) -> np.ndarray:
        ann = self.root / "Annotations" / video / (frame_path.stem + ".png")
        if not ann.exists():
            return np.zeros((1, 1), dtype=np.uint8)
        m = np.array(Image.open(ann))
        if m.ndim == 3:
            m = m[..., 0]
        return (m == obj_id).astype(np.uint8)

    def __getitem__(self, idx: int) -> dict:
        video = self.videos[idx]
        frames = self._list_frames(video)
        if len(frames) < self.clip_len:
            # Repeat the last frame to pad short videos.
            frames = frames + [frames[-1]] * (self.clip_len - len(frames))

        # Sample stride and start so the clip fits.
        for _ in range(4):
            stride = self.rng.choice(self.stride_choices)
            span = stride * (self.clip_len - 1)
            if span < len(frames):
                break
        start = self.rng.randint(0, max(0, len(frames) - span - 1))
        idxs = [start + i * stride for i in range(self.clip_len)]
        idxs = [min(i, len(frames) - 1) for i in idxs]
        clip_paths = [frames[i] for i in idxs]

        # Pick up to `objects_per_clip` object ids present in the first sampled frame.
        ann0 = np.array(
            Image.open(
                self.root / "Annotations" / video / (clip_paths[0].stem + ".png")
            )
        )
        if ann0.ndim == 3:
            ann0 = ann0[..., 0]
        present_ids = [int(i) for i in np.unique(ann0) if i != 0]
        if not present_ids:
            # Degenerate clip — return a single no-object example so loss masking handles it.
            obj_ids = [1]
        elif len(present_ids) <= self.objects_per_clip:
            obj_ids = present_ids
        else:
            obj_ids = self.rng.sample(present_ids, self.objects_per_clip)

        # Shared spatial augmentation params for the whole clip.
        first = (
            torch.from_numpy(np.array(Image.open(clip_paths[0]).convert("RGB")))
            .permute(2, 0, 1)
            .float()
            / 255.0
        )
        params = sample_params(
            self.image_size,
            src_h=first.shape[-2],
            src_w=first.shape[-1],
            scale_range=self.scale_range,
            hflip_prob=self.hflip_prob,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            grayscale_prob=self.grayscale_prob,
            affine_degree=self.affine_degree,
            affine_shear=self.affine_shear,
            rng=self.rng,
        )

        # Load and augment each frame; record per-object masks per frame.
        frames_t: list[torch.Tensor] = []
        masks_per_obj: list[list[torch.Tensor]] = [[] for _ in obj_ids]
        for fp in clip_paths:
            img_raw = (
                torch.from_numpy(np.array(Image.open(fp).convert("RGB")))
                .permute(2, 0, 1)
                .float()
                / 255.0
            )
            img_aug = None
            for oi, obj_id in enumerate(obj_ids):
                mask_np = self._load_mask(video, fp, obj_id)
                mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)
                im_out, m_out = apply_image(
                    img_raw, mask, params, self.image_size, apply_color=True
                )
                if img_aug is None:
                    img_aug = im_out
                masks_per_obj[oi].append((m_out > 0.5).float())
            frames_t.append(normalize(img_aug))
        frames_t = torch.stack(frames_t, dim=0)  # [T, 3, H, W]
        masks_KT = torch.stack(
            [torch.stack(seq, dim=0) for seq in masks_per_obj], dim=0
        )  # [K, T, 1, H, W]

        has_object = (
            masks_KT.flatten(2).sum(-1) >= self.min_mask_pixels
        ).float()  # [K, T]

        frame0_prompts = [
            self.prompt_sampler.sample(masks_KT[k, 0])
            if has_object[k, 0] > 0
            else PromptSample(
                point_coords=torch.zeros(1, 2),
                point_labels=torch.tensor([-1], dtype=torch.int32),
            )
            for k in range(len(obj_ids))
        ]
        # Same correction-frame schedule across objects in the clip (simpler bookkeeping).
        correction_frames: list[int] = []
        if self.correction_prob > 0:
            for t in range(1, self.clip_len):
                if has_object[:, t].any() and self.rng.random() < self.correction_prob:
                    correction_frames.append(t)
                if len(correction_frames) >= self.max_correction_frames:
                    break

        return {
            "frames": frames_t,  # [T, 3, H, W]
            "gt_masks": masks_KT,  # [K, T, 1, H, W]
            "has_object": has_object,  # [K, T]
            "frame0_prompts": frame0_prompts,  # list[K] of PromptSample
            "correction_frames": correction_frames,
            "video_id": video,
        }


def collate_video_batch(batch: list[dict]) -> dict:
    """Flatten the object dim into the batch dim so the model sees `[N, T, ...]`.

    Each clip contributes K rows (one per tracked object). The clip's frames are
    repeated K times along the new batch dim — the image encoder will redundantly
    process them. For paper-faithful object counts (3 per video frame), set
    `objects_per_clip` accordingly in the training config.
    """
    frames_rows: list[torch.Tensor] = []
    masks_rows: list[torch.Tensor] = []
    has_rows: list[torch.Tensor] = []
    prompts_flat: list = []
    correction: list[list[int]] = []
    video_ids: list[str] = []
    for b in batch:
        K = b["gt_masks"].shape[0]
        # Repeat frames K times so each tracked object has its own row.
        frames_rows.extend([b["frames"]] * K)
        for k in range(K):
            masks_rows.append(b["gt_masks"][k])  # [T, 1, H, W]
            has_rows.append(b["has_object"][k])  # [T]
            prompts_flat.append(b["frame0_prompts"][k])
            correction.append(list(b["correction_frames"]))
            video_ids.append(b["video_id"])
    frames = torch.stack(frames_rows, dim=0)  # [N, T, 3, H, W]
    masks = torch.stack(masks_rows, dim=0)  # [N, T, 1, H, W]
    has_object = torch.stack(has_rows, dim=0)  # [N, T]

    max_pts = max(
        (p.point_coords.shape[0] if p.point_coords is not None else 1)
        for p in prompts_flat
    )
    N = len(prompts_flat)
    coords = torch.zeros(N, max_pts, 2)
    labels = -torch.ones(N, max_pts, dtype=torch.int32)
    for i, p in enumerate(prompts_flat):
        if p.point_coords is not None:
            n = p.point_coords.shape[0]
            coords[i, :n] = p.point_coords
            labels[i, :n] = p.point_labels

    return {
        "frames": frames,
        "gt_masks": masks,
        "has_object": has_object,
        "frame0_point_inputs": {"point_coords": coords, "point_labels": labels},
        "correction_frames": correction,
        "video_ids": video_ids,
    }
