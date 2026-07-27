"""Torch FLAME model."""

from pathlib import Path
from typing import Literal

from torch import nn

from body_models._rotations import RotationType
from body_models._runtime import TorchRuntime
from body_models.flame._model import FLAMEModel


class FLAME(FLAMEModel, nn.Module):
    """FLAME using Torch tensors and optional Warp kernels."""

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
