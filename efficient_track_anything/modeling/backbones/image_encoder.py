# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficient_track_anything.modeling.efficienttam_utils import LayerNorm2d


EFFICIENTSAM_PRETRAINED_URLS = {
    "efficient_sam_vitt": (
        "https://huggingface.co/yunyangx/EfficientSAM/resolve/main/"
        "efficient_sam_vitt.pt"
    ),
    "efficient_sam_vits": (
        "https://huggingface.co/yunyangx/EfficientSAM/resolve/main/"
        "efficient_sam_vits.pt"
    ),
}


def _load_pretrained_state(weights_path: str) -> dict:
    if weights_path in EFFICIENTSAM_PRETRAINED_URLS:
        weights_path = EFFICIENTSAM_PRETRAINED_URLS[weights_path]

    if weights_path.startswith(("http://", "https://")):
        payload = torch.hub.load_state_dict_from_url(
            weights_path, map_location="cpu", progress=True
        )
    else:
        path = Path(weights_path).expanduser()
        payload = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def _map_efficientsam_image_encoder_key(key: str) -> str | None:
    if key.startswith("module."):
        key = key[len("module.") :]
    if not key.startswith("image_encoder."):
        return None

    key = key[len("image_encoder.") :]
    if (
        key.startswith("patch_embed.")
        or key == "pos_embed"
        or key.startswith("blocks.")
    ):
        key = key.replace(".mlp.fc1.", ".mlp.layers.0.")
        key = key.replace(".mlp.fc2.", ".mlp.layers.1.")
        return f"trunk.{key}"

    neck_map = {
        "neck.0.": "neck.convs.0.conv_1x1.",
        "neck.1.": "neck.convs.0.norm_0.",
        "neck.2.": "neck.convs.0.conv_3x3.",
        "neck.3.": "neck.convs.0.norm_1.",
    }
    for src, dst in neck_map.items():
        if key.startswith(src):
            return f"{dst}{key[len(src) :]}"
    return None


class ImageEncoder(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
        weights_path: Optional[str] = None,
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert self.trunk.channel_list == self.neck.backbone_channel_list, (
            f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"
        )
        if weights_path is not None:
            self._load_efficientsam_image_encoder_weights(weights_path)

    def _load_efficientsam_image_encoder_weights(self, weights_path: str) -> None:
        source_state = _load_pretrained_state(weights_path)
        target_state = self.state_dict()
        mapped_state = {}
        skipped_shape = []

        for key, value in source_state.items():
            mapped_key = _map_efficientsam_image_encoder_key(key)
            if mapped_key is None or mapped_key not in target_state:
                continue
            if value.shape != target_state[mapped_key].shape:
                skipped_shape.append(
                    (
                        mapped_key,
                        tuple(value.shape),
                        tuple(target_state[mapped_key].shape),
                    )
                )
                continue
            mapped_state[mapped_key] = value

        if not mapped_state:
            raise RuntimeError(
                f"No compatible EfficientSAM image-encoder weights found in {weights_path}"
            )

        self.load_state_dict(mapped_state, strict=False)
        print(
            f"[ImageEncoder] loaded {len(mapped_state)}/{len(target_state)} "
            f"image-encoder params from {weights_path}"
        )
        if skipped_shape:
            preview = ", ".join(
                f"{name}: {src}->{dst}" for name, src, dst in skipped_shape[:3]
            )
            suffix = "..." if len(skipped_shape) > 3 else ""
            print(
                f"[ImageEncoder] skipped {len(skipped_shape)} shape-mismatched "
                f"params ({preview}{suffix})"
            )

    def forward(self, sample: torch.Tensor):
        # Forward through backbone
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


class ViTDetNeck(nn.Module):
    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        neck_norm=None,
    ):
        """Initialize the neck

        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.backbone_channel_list = backbone_channel_list
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.d_model = d_model
        use_bias = neck_norm is None
        for dim in self.backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            if neck_norm is not None:
                current.add_module("norm_0", LayerNorm2d(d_model))
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            if neck_norm is not None:
                current.add_module("norm_1", LayerNorm2d(d_model))
            self.convs.append(current)

    def forward(self, xs: List[torch.Tensor]):
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)

        x = xs[0]
        x_out = self.convs[0](x)
        out[0] = x_out
        pos[0] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos
