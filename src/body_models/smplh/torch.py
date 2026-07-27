"""Torch SMPL-H model."""

from pathlib import Path
from typing import Literal

from torch import nn

from body_models._rotations import RotationType
from body_models._runtime import TorchRuntime
from body_models.smplh._model import SMPLHModel


class SMPLH(SMPLHModel, nn.Module):
    """SMPL-H using Torch tensors and optional Warp kernels."""

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
        SMPLHModel.__init__(
            self,
            model_path,
            gender,
            flat_hand_mean,
            simplify,
            rotation_type,
            runtime=TorchRuntime(skinning_backend),
        )


__all__ = ["SMPLH"]
