"""Torch FLAME model."""

from pathlib import Path
from typing import Literal

import torch.nn as nn

from body_models.parts.flame.model import FLAMEModel
from body_models.rotations import RotationType
from body_models.runtime import TorchRuntime


class FLAME(FLAMEModel, nn.Module):
    """FLAME using Torch tensors and optional Warp kernels."""

    skinning_backends = TorchRuntime.skinning_backends

    def __init__(
        self,
        model_path: Path | str | None = None,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        skinning_backend: Literal["torch", "warp"] = "torch",
    ) -> None:
        nn.Module.__init__(self)
        FLAMEModel.__init__(
            self,
            model_path,
            simplify,
            rotation_type,
            runtime=TorchRuntime(skinning_backend),
        )


__all__ = ["FLAME"]
