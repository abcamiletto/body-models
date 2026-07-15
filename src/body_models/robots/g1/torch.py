"""Torch Unitree G1 model."""

from pathlib import Path

import torch.nn as nn

from body_models.robots.g1 import core
from body_models.robots.g1.model import G1Model
from body_models.runtime import TorchRuntime


class G1(G1Model, nn.Module):
    """Unitree G1 using Torch tensors."""

    kernels = ("torch",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        convention: core.Convention = "soma",
    ) -> None:
        nn.Module.__init__(self)
        G1Model.__init__(self, model_path, convention=convention, runtime=TorchRuntime())


__all__ = ["G1"]
