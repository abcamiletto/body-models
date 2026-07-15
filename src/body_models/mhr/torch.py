"""Torch MHR model."""

from pathlib import Path
from typing import Literal

import torch.nn as nn

from body_models.bodies.mhr.model import MHRModel
from body_models.runtime import TorchRuntime


class MHR(MHRModel, nn.Module):
    """MHR using Torch tensors and optional Warp kernels."""

    skinning_backends = TorchRuntime.skinning_backends

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        lod: int = 1,
        simplify: float = 1.0,
        skinning_backend: Literal["torch", "warp"] = "torch",
    ) -> None:
        nn.Module.__init__(self)
        MHRModel.__init__(
            self,
            model_path,
            lod=lod,
            simplify=simplify,
            runtime=TorchRuntime(skinning_backend),
        )


__all__ = ["MHR"]
