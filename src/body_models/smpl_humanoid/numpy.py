"""NumPy SMPL humanoid model."""

from pathlib import Path

from body_models.smpl_humanoid.model import SmplHumanoidModel
from body_models.runtime import NumpyRuntime


class SmplHumanoid(SmplHumanoidModel):
    """SMPL humanoid using NumPy arrays."""

    def __init__(self, source: Path | str = "humenv") -> None:
        super().__init__(source, runtime=NumpyRuntime())


__all__ = ["SmplHumanoid"]
