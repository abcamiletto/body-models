"""Torch SMPL-H model."""

from pathlib import Path
from typing import Literal

import torch.nn as nn

from body_models.bodies.smplh.model import SMPLHModel
from body_models.rotations import RotationType
from body_models.runtime import TorchRuntime


class SMPLH(SMPLHModel, nn.Module):
    """SMPL-H using Torch tensors and optional Warp kernels."""

    kernels = TorchRuntime.kernels

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        kernel: Literal["torch", "warp"] = "torch",
    ) -> None:
        nn.Module.__init__(self)
        SMPLHModel.__init__(
            self,
            model_path,
            gender,
            flat_hand_mean,
            simplify,
            rotation_type,
            runtime=TorchRuntime(kernel),
        )


__all__ = ["SMPLH"]
