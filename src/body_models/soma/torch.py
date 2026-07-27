"""Torch SOMA model."""

from pathlib import Path
from typing import Literal

from torch import nn

from body_models.rotations import RotationType
from body_models.runtime import TorchRuntime
from body_models.soma.model import SOMAModel


class SOMA(SOMAModel, nn.Module):
    """SOMA using Torch tensors and optional Warp skinning."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        model_type: str = "soma",
        lod: str = "mid",
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        skinning_backend: Literal["torch", "warp"] = "torch",
    ) -> None:
        nn.Module.__init__(self)
        SOMAModel.__init__(
            self,
            model_path,
            model_type=model_type,
            lod=lod,
            simplify=simplify,
            rotation_type=rotation_type,
            runtime=TorchRuntime(skinning_backend),
        )


__all__ = ["SOMA"]
