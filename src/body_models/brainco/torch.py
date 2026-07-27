"""Torch BrainCo Revo 2 model."""

from pathlib import Path

from torch import nn

from body_models._runtime import TorchRuntime
from body_models.brainco._io import Side
from body_models.brainco._model import BrainCoHandModel


class BrainCoHand(BrainCoHandModel, nn.Module):
    """BrainCo Revo 2 using Torch tensors."""

    def __init__(self, model_path: Path | str | None = None, *, side: Side = "right") -> None:
        nn.Module.__init__(self)
        BrainCoHandModel.__init__(
            self,
            model_path,
            side=side,
            runtime=TorchRuntime(),
        )


__all__ = ["BrainCoHand"]
