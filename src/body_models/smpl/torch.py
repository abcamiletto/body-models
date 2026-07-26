"""Torch SMPL model."""

from pathlib import Path
from typing import Literal

from torch import nn

from body_models.rotations import RotationType
from body_models.runtime import TorchRuntime
from body_models.smpl.model import SMPLModel


class SMPL(SMPLModel, nn.Module):
    """SMPL using Torch tensors and optional Warp kernels."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        skinning_backend: Literal["torch", "warp"] = "torch",
    ) -> None:
        nn.Module.__init__(self)
        SMPLModel.__init__(
            self,
            model_path,
            gender,
            simplify,
            rotation_type,
            runtime=TorchRuntime(skinning_backend),
        )


__all__ = ["SMPL"]
