"""Image / video augmentations matching the EfficientTAM paper recipe.

Paper §experimental setup: horizontal flip, affine (degree=25, shear=20),
color jitter (brightness=0.1, contrast=0.03, saturation=0.03), and grayscale (p=0.05).

For video, augmentation params are drawn once per clip and applied identically
to every frame so spatial correspondence is preserved across the clip.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torchvision.transforms.v2.functional as TF


@dataclass
class AugmentParams:
    hflip: bool = False
    crop: tuple[int, int, int, int] | None = None  # (top, left, h, w) in source pixels
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    grayscale: bool = False
    affine_angle: float = 0.0  # rotation degrees
    affine_shear: tuple[float, float] = (0.0, 0.0)  # (shear_x, shear_y) degrees


def sample_params(
    image_size: int,
    src_h: int,
    src_w: int,
    scale_range: tuple[float, float] = (0.5, 1.0),
    hflip_prob: float = 0.5,
    brightness: float = 0.1,
    contrast: float = 0.03,
    saturation: float = 0.03,
    grayscale_prob: float = 0.05,
    affine_degree: float = 25.0,
    affine_shear: float = 20.0,
    rng: random.Random | None = None,
) -> AugmentParams:
    rng = rng or random
    scale = rng.uniform(*scale_range)
    crop_h = max(8, int(min(src_h, src_w) * scale))
    crop_w = crop_h
    top = rng.randint(0, max(0, src_h - crop_h))
    left = rng.randint(0, max(0, src_w - crop_w))
    return AugmentParams(
        hflip=(rng.random() < hflip_prob),
        crop=(top, left, crop_h, crop_w),
        brightness=1.0 + rng.uniform(-brightness, brightness),
        contrast=1.0 + rng.uniform(-contrast, contrast),
        saturation=1.0 + rng.uniform(-saturation, saturation),
        grayscale=(rng.random() < grayscale_prob),
        affine_angle=rng.uniform(-affine_degree, affine_degree),
        affine_shear=(
            rng.uniform(-affine_shear, affine_shear),
            rng.uniform(-affine_shear, affine_shear),
        ),
    )


def apply_image(
    img: torch.Tensor,  # [3, H, W], float in [0, 1] or uint8
    mask: torch.Tensor,  # [1, H, W], binary float
    params: AugmentParams,
    out_size: int,
    apply_color: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if params.crop is not None:
        top, left, h, w = params.crop
        img = TF.crop(img, top, left, h, w)
        mask = TF.crop(mask, top, left, h, w)
    img = TF.resize(img, [out_size, out_size], antialias=True)
    mask = TF.resize(
        mask, [out_size, out_size], interpolation=TF.InterpolationMode.NEAREST
    )
    # Affine: rotation + shear. Translate/scale are handled by the crop+resize above.
    if abs(params.affine_angle) > 1e-3 or any(
        abs(s) > 1e-3 for s in params.affine_shear
    ):
        img = TF.affine(
            img,
            angle=params.affine_angle,
            translate=[0, 0],
            scale=1.0,
            shear=list(params.affine_shear),
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        mask = TF.affine(
            mask,
            angle=params.affine_angle,
            translate=[0, 0],
            scale=1.0,
            shear=list(params.affine_shear),
            interpolation=TF.InterpolationMode.NEAREST,
        )
    if params.hflip:
        img = TF.hflip(img)
        mask = TF.hflip(mask)
    if apply_color and img.dtype.is_floating_point:
        img = TF.adjust_brightness(img, params.brightness)
        img = TF.adjust_contrast(img, params.contrast)
        img = TF.adjust_saturation(img, params.saturation)
        if params.grayscale:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)
    return img, mask


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalize(img: torch.Tensor) -> torch.Tensor:
    """ImageNet normalization, matching `EfficientTAMTransforms`."""
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    return (img - _IMAGENET_MEAN.to(img.device)) / _IMAGENET_STD.to(img.device)
