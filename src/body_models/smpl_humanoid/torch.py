"""Torch SMPL humanoid model."""

from pathlib import Path

import torch.nn as nn

from body_models.robots.smpl_humanoid.model import SmplHumanoidModel
from body_models.runtime import TorchRuntime
from body_models.state import torch_state


class SmplHumanoid(SmplHumanoidModel, nn.Module):
    """SMPL humanoid using Torch tensors."""

    skinning_backends = ("torch",)

    def __init__(self, source: Path | str = "humenv") -> None:
        nn.Module.__init__(self)
        SmplHumanoidModel.__init__(
            self,
            source,
            runtime=TorchRuntime(),
            materialize=torch_state,
        )


__all__ = ["SmplHumanoid"]
