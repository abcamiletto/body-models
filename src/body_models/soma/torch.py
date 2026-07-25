"""Torch SOMA model."""

from pathlib import Path
from typing import Literal

import torch.nn as nn

from body_models.soma import correctives_torch
from body_models.soma.model import SOMAModel
from body_models.rotations import RotationType
from body_models.runtime import TorchRuntime

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
        match_warp: bool = True,
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
            match_warp=match_warp,
            runtime=TorchRuntime(skinning_backend),
            corrective_network=correctives_torch.TorchCorrectiveNetwork,
        )


__all__ = ["SOMA"]
