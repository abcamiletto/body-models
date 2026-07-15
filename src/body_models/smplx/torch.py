"""Torch SMPL-X model."""

from pathlib import Path
from typing import Literal

import torch.nn as nn

from body_models.bodies.smplx.model import SMPLXModel
from body_models.rotations import RotationType
from body_models.runtime import TorchRuntime


class SMPLX(SMPLXModel, nn.Module):
    """SMPL-X using Torch tensors and optional Warp kernels."""

    skinning_backends = TorchRuntime.skinning_backends

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        skinning_backend: Literal["torch", "warp"] = "torch",
    ) -> None:
        nn.Module.__init__(self)
        SMPLXModel.__init__(
            self,
            model_path,
            gender,
            flat_hand_mean,
            simplify,
            rotation_type,
            runtime=TorchRuntime(skinning_backend),
        )


__all__ = ["SMPLX"]
