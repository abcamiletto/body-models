"""Torch ANNY model."""

from pathlib import Path
from typing import Literal

import torch.nn as nn

from body_models.bodies.anny.model import ANNYModel
from body_models.rotations import RotationType
from body_models.runtime import TorchRuntime


class ANNY(ANNYModel, nn.Module):
    """ANNY using Torch tensors and optional Warp kernels."""

    skinning_backends = TorchRuntime.skinning_backends

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rig: str = "default",
        topology: str = "default",
        all_phenotypes: bool = False,
        extrapolate_phenotypes: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        skinning_backend: Literal["torch", "warp"] = "torch",
    ) -> None:
        nn.Module.__init__(self)
        ANNYModel.__init__(
            self,
            model_path,
            rig=rig,
            topology=topology,
            all_phenotypes=all_phenotypes,
            extrapolate_phenotypes=extrapolate_phenotypes,
            simplify=simplify,
            rotation_type=rotation_type,
            runtime=TorchRuntime(skinning_backend),
        )


__all__ = ["ANNY"]
