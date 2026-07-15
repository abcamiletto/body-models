"""Torch SOMA model."""

from pathlib import Path
from typing import Literal

import torch.nn as nn

from body_models.bodies.soma.model import SOMAModel
from body_models.rotations import RotationType
from body_models.runtime import TorchRuntime


class SOMA(SOMAModel, nn.Module):
    """SOMA using Torch tensors and optional Warp skinning."""

    kernels = TorchRuntime.kernels

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        model_type: str = "soma",
        lod: str = "mid",
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        match_warp: bool = True,
        kernel: Literal["torch", "warp"] = "torch",
    ) -> None:
        nn.Module.__init__(self)
        SOMAModel.__init__(
            self,
            model_path,
            model_type=model_type,
            lod=lod,
            simplify=simplify,
            rotation_type=rotation_type,
            match_warp=match_warp,
            runtime=TorchRuntime(kernel),
        )


__all__ = ["SOMA"]
