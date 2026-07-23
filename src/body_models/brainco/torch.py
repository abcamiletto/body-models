"""Torch BrainCo Revo 2 model."""

from pathlib import Path

import torch.nn as nn

from body_models.robots.brainco.io import Side
from body_models.robots.brainco.model import BrainCoHandModel
from body_models.runtime import TorchRuntime
from body_models.state import torch_state


class BrainCoHand(BrainCoHandModel, nn.Module):
    """BrainCo Revo 2 using Torch tensors."""

    skinning_backends = ("torch",)

    def __init__(self, model_path: Path | str | None = None, *, side: Side = "right") -> None:
        nn.Module.__init__(self)
        BrainCoHandModel.__init__(
            self,
            model_path,
            side=side,
            runtime=TorchRuntime(),
            materialize=torch_state,
        )


__all__ = ["BrainCoHand"]
