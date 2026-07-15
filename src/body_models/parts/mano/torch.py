"""Torch MANO model."""

from pathlib import Path
from typing import Literal

import torch.nn as nn

from body_models.parts.mano.model import MANOModel
from body_models.rotations import RotationType
from body_models.runtime import TorchRuntime


class MANO(MANOModel, nn.Module):
    """MANO using Torch tensors and optional Warp kernels."""

    kernels = TorchRuntime.kernels

    def __init__(
        self,
        model_path: Path | str | None = None,
        side: Literal["right", "left"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        kernel: Literal["torch", "warp"] = "torch",
    ) -> None:
        nn.Module.__init__(self)
        MANOModel.__init__(
            self,
            model_path,
            side,
            flat_hand_mean,
            simplify,
            rotation_type,
            runtime=TorchRuntime(kernel),
        )


__all__ = ["MANO"]
