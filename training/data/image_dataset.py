"""Image segmentation dataset for Stage 1 (SA-1B-style folder layout).

Expected layout:

    {root}/
        images/
            {id}.jpg
        masks/
            {id}.png         # binary PNG, single object per file

Multi-object cases are flattened: emit `{id}_{obj}.png` for each object.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from training.data.augment import AugmentParams, apply_image, normalize, sample_params
from training.data.prompts import PromptSample, PromptSampler


def _index_images(root: Path) -> list[tuple[Path, list[Path]]]:
    """Build `[(image_path, [mask_path, ...]), ...]` — one entry per image, all masks listed."""
    image_dir = root / "images"
    mask_dir = root / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Expected {image_dir} and {mask_dir} to exist for SA-1B-style layout"
        )
    image_to_masks: dict[str, list[Path]] = {}
    for mp in sorted(mask_dir.glob("*.png")):
        stem = mp.stem
        # Mask {id}_{obj}.png maps to image {id}; mask {id}.png also maps to {id}
        img_stem = (
            stem.rsplit("_", 1)[0]
            if "_" in stem and stem.rsplit("_", 1)[1].isdigit()
            else stem
        )
        image_to_masks.setdefault(img_stem, []).append(mp)
    entries: list[tuple[Path, list[Path]]] = []
    for img_stem, masks in sorted(image_to_masks.items()):
        ip = image_dir / f"{img_stem}.jpg"
        if not ip.exists():
            ip = image_dir / f"{img_stem}.png"
        if ip.exists():
            entries.append((ip, masks))
    return entries


class ImageSegmentationDataset(Dataset):
    """Yields one (image, single-object mask, prompt) triplet per sample."""

    def __init__(
        self,
        root: str | os.PathLike,
        image_size: int,
        prompt_sampler: PromptSampler,
        scale_range: tuple[float, float] = (0.5, 1.0),
        hflip_prob: float = 0.5,
        brightness: float = 0.1,
        contrast: float = 0.03,
        saturation: float = 0.03,
        grayscale_prob: float = 0.05,
        affine_degree: float = 25.0,
        affine_shear: float = 20.0,
        min_mask_pixels: int = 32,
        objects_per_image: int = 1,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.prompt_sampler = prompt_sampler
        self.scale_range = scale_range
        self.hflip_prob = hflip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.grayscale_prob = grayscale_prob
        self.affine_degree = affine_degree
        self.affine_shear = affine_shear
        self.min_mask_pixels = min_mask_pixels
        self.objects_per_image = objects_per_image
        self.entries = _index_images(self.root)
        if not self.entries:
            raise RuntimeError(f"No (image, masks) entries found under {self.root}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        ip, mask_paths = self.entries[idx]
        img = (
            torch.from_numpy(np.array(Image.open(ip).convert("RGB")))
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        # Pick up to `objects_per_image` masks from those available.
        import random as _rand

        if len(mask_paths) > self.objects_per_image:
            mask_paths = _rand.sample(mask_paths, self.objects_per_image)
        masks_np = []
        for mp in mask_paths:
            arr = np.array(Image.open(mp))
            if arr.ndim == 3:
                arr = arr[..., 0]
            masks_np.append((arr > 0).astype(np.float32))
        # Shared spatial augmentation: same crop/affine/flip for image and all masks.
        params = sample_params(
            self.image_size,
            src_h=img.shape[-2],
            src_w=img.shape[-1],
            scale_range=self.scale_range,
            hflip_prob=self.hflip_prob,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            grayscale_prob=self.grayscale_prob,
            affine_degree=self.affine_degree,
            affine_shear=self.affine_shear,
        )
        # Apply image augmentation once; reuse the augmented mask routine per object.
        # We bundle one mask through apply_image to get the augmented image, then run
        # apply_image again per additional mask using the same params.
        aug_img, aug_masks = None, []
        for m_np in masks_np:
            mask = torch.from_numpy(m_np).unsqueeze(0)
            im_out, m_out = apply_image(
                img, mask, params, self.image_size, apply_color=True
            )
            if aug_img is None:
                aug_img = im_out
            aug_masks.append((m_out > 0.5).float())
        aug_img = normalize(aug_img)
        masks = torch.stack(aug_masks, dim=0)  # [K, 1, H, W]

        has_object = (masks.flatten(1).sum(1) >= self.min_mask_pixels).float()  # [K]
        prompts = [
            self.prompt_sampler.sample(masks[k])
            if has_object[k] > 0
            else PromptSample()
            for k in range(masks.shape[0])
        ]

        return {
            "image": aug_img,  # [3, H, W] normalized
            "gt_masks": masks,  # [K, 1, H, W] in {0, 1}
            "has_object": has_object,  # [K]
            "prompts": prompts,  # list[K] of PromptSample
        }


def collate_image_batch(batch: list[dict]) -> dict:
    """Flatten object axis into the batch dim so the model sees `[B*K_total, ...]`.

    Each sample contributes its own image repeated K times (where K is the number
    of objects sampled from that image). The image encoder will redundantly
    process the duplicates — that is the throughput cost of multi-object training
    on a single GPU.
    """
    images_out: list[torch.Tensor] = []
    masks_out: list[torch.Tensor] = []
    has_out: list[torch.Tensor] = []
    prompts_flat: list = []
    for b in batch:
        K = b["gt_masks"].shape[0]
        images_out.extend([b["image"]] * K)
        masks_out.append(b["gt_masks"])
        has_out.append(b["has_object"])
        prompts_flat.extend(b["prompts"])
    images = torch.stack(images_out, dim=0)  # [N=sum_k, 3, H, W]
    masks = torch.cat(masks_out, dim=0)  # [N, 1, H, W]
    has = torch.cat(has_out, dim=0)  # [N]

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
        "image": images,
        "gt_mask": masks,
        "has_object": has,
        "point_inputs": {"point_coords": coords, "point_labels": labels},
    }
