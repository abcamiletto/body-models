"""Torch SMPL humanoid model."""

from pathlib import Path

from torch import nn

from body_models.runtime import TorchRuntime
from body_models.smpl_humanoid.model import SmplHumanoidModel


class SmplHumanoid(SmplHumanoidModel, nn.Module):
    """SMPL humanoid using Torch tensors."""

    def __init__(self, source: Path | str = "humenv") -> None:
        nn.Module.__init__(self)
        SmplHumanoidModel.__init__(
            self,
            source,
            runtime=TorchRuntime(),
        )


__all__ = ["SmplHumanoid"]
