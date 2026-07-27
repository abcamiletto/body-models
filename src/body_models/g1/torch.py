"""Torch Unitree G1 model."""

from pathlib import Path

from torch import nn

from body_models._runtime import TorchRuntime
from body_models.g1 import _core as core
from body_models.g1._model import G1Model


class G1(G1Model, nn.Module):
    """Unitree G1 using Torch tensors."""

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
        )


__all__ = ["G1"]
