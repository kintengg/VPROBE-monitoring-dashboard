import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.core import register


try:
    from fastervit.models import (
        faster_vit_0_any_res,
        faster_vit_1_any_res,
        faster_vit_2_any_res,
        faster_vit_3_any_res,
        faster_vit_4_any_res,
        faster_vit_5_any_res,
        faster_vit_6_any_res,
    )
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "FasterViTBackbone requires the `fastervit` package. "
        "Install it via `pip install fastervit`."
    ) from exc


_VARIANT_FACTORIES = {
    "fastervit-0": faster_vit_0_any_res,
    "fastervit-1": faster_vit_1_any_res,
    "fastervit-2": faster_vit_2_any_res,
    "fastervit-3": faster_vit_3_any_res,
    "fastervit-4": faster_vit_4_any_res,
    "fastervit-5": faster_vit_5_any_res,
    "fastervit-6": faster_vit_6_any_res,
}

# Channel dimensions (stage1, stage2, stage4) for each any-res FasterViT variant.
_STAGE_CHANNELS: Dict[str, Tuple[int, int, int, int]] = {
    "fastervit-0": (128, 256, 512, 512),
    "fastervit-1": (160, 320, 640, 640),
    "fastervit-2": (192, 384, 768, 768),
    "fastervit-3": (256, 512, 1024, 1024),
    "fastervit-4": (392, 784, 1568, 1568),
    "fastervit-5": (640, 1280, 2560, 2560),
    "fastervit-6": (640, 1280, 2560, 2560),
}


def _to_hw(resolution: Sequence[int]) -> Tuple[int, int]:
    if len(resolution) != 2:
        raise ValueError("resolution must be a sequence of length 2 (H, W).")
    return int(resolution[0]), int(resolution[1])


@register
class FasterViTBackbone(nn.Module):
    """Adapter that exposes FasterViT any-resolution variants as RT-DETR backbones.

    The module returns the outputs of stages 1, 2, and 4 (1-indexed) from the
    selected FasterViT variant, matching strides (4, 8, 32) for 640x640 inputs by
    default. The final stage output is normalized via the model's head norm to
    mirror the native FasterViT forward pass.
    """

    def __init__(
        self,
        variant: str = "fastervit-0",
        resolution: Sequence[int] | int = (640, 640),
        pretrained: bool = False,
        return_stages: Sequence[int] = (1, 2, 4),
        apply_last_norm: bool = True,
        model_path: Optional[str] = None,
        **factory_kwargs: Any,
    ) -> None:
        super().__init__()

        if isinstance(resolution, int):  # allow single value for square inputs
            resolution = (resolution, resolution)
        self.resolution = _to_hw(resolution)

        if variant not in _VARIANT_FACTORIES:
            raise ValueError(f"Unsupported FasterViT variant '{variant}'.")

        self.variant = variant
        self.factory_kwargs = dict(factory_kwargs)
        if model_path is not None:
            self.factory_kwargs.setdefault("model_path", model_path)
        self.model_path = model_path
        self.backbone = _VARIANT_FACTORIES[variant](
            pretrained=pretrained,
            resolution=self.resolution,
            **self.factory_kwargs,
        )
        self.return_indices = sorted({stage - 1 for stage in return_stages})
        if not self.return_indices:
            raise ValueError("return_stages must contain at least one stage index.")
        if max(self.return_indices) >= len(self.backbone.levels):
            raise ValueError("Requested stage index exceeds FasterViT depth.")

        unknown = set(self.return_indices) - {0, 1, 2, 3}
        if unknown:
            raise ValueError(f"FasterViT exposes exactly 4 stages. Invalid indices: {unknown}")

        if (3 in self.return_indices) and len(self.return_indices) > 3:
            warnings.warn(
                "Using all FasterViT stages may increase memory usage considerably.",
                RuntimeWarning,
            )

        channel_tuple = _STAGE_CHANNELS[variant]
        # Map requested stage order to corresponding channel dimensions.
        stage_to_channel = {i: channel_tuple[i] for i in range(len(channel_tuple))}
        missing = set(self.return_indices) - set(stage_to_channel)
        if missing:
            raise ValueError(f"Unsupported stage indices requested: {sorted(missing)}")
        self.out_channels: List[int] = [stage_to_channel[idx] for idx in self.return_indices]

        # FasterViT strides for 640x640 inputs: stage1=8, stage2=16, stage3/4=32.
        # We derive stride dynamically based on resolution but keep canonical defaults here.
        stride_map = {0: 8, 1: 16, 2: 32, 3: 32}
        self.out_strides: List[int] = [stride_map[idx] for idx in self.return_indices]

        self.apply_last_norm = apply_last_norm
        self.return_idx = self.return_indices

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        features = self.backbone.patch_embed(x)
        for idx, level in enumerate(self.backbone.levels):
            features = level(features)
            if idx in self.return_indices:
                feats.append(features)
        # Apply the terminal norm if we returned the final stage.
        if self.apply_last_norm and self.return_indices and self.return_indices[-1] == len(self.backbone.levels) - 1:
            feats[-1] = self.backbone.norm(feats[-1])
        return feats

    def to(self, *args: Any, **kwargs: Any) -> "FasterViTBackbone":
        super().to(*args, **kwargs)
        self.backbone.to(*args, **kwargs)
        return self

    def eval(self) -> "FasterViTBackbone":
        self.backbone.eval()
        return super().eval()

    def train(self, mode: bool = True) -> "FasterViTBackbone":
        self.backbone.train(mode)
        return super().train(mode)

    @property
    def stages(self) -> Iterable[nn.Module]:  # pragma: no cover - helper for debugging
        return self.backbone.levels
