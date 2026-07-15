"""Torch SKEL model."""

from pathlib import Path
from typing import Literal

import torch.nn as nn

from body_models.runtime import TorchRuntime
from body_models.skeletons.skel.model import SKELModel


class SKEL(SKELModel, nn.Module):
    """SKEL using Torch tensors and optional Warp kernels."""

    skinning_backends = TorchRuntime.skinning_backends

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["male", "female"] | None = None,
        simplify: float = 1.0,
        skinning_backend: Literal["torch", "warp"] = "torch",
    ) -> None:
        nn.Module.__init__(self)
        SKELModel.__init__(
            self,
            model_path,
            gender,
            simplify,
            runtime=TorchRuntime(skinning_backend),
        )


__all__ = ["SKEL"]
