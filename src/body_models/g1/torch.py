"""Torch Unitree G1 model."""

from pathlib import Path

import torch.nn as nn

from body_models.robots.g1 import core
from body_models.robots.g1.model import G1Model
from body_models.runtime import TorchRuntime
from body_models.state import torch_state


class G1(G1Model, nn.Module):
    """Unitree G1 using Torch tensors."""

    skinning_backends = ("torch",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        convention: core.Convention = "soma",
    ) -> None:
        nn.Module.__init__(self)
        G1Model.__init__(
            self,
            model_path,
            convention=convention,
            runtime=TorchRuntime(),
            materialize=torch_state,
        )


__all__ = ["G1"]
