"""NumPy SMPL humanoid model."""

from pathlib import Path

from body_models._runtime import NumpyRuntime
from body_models.smpl_humanoid._model import SmplHumanoidModel


class SmplHumanoid(SmplHumanoidModel):
    """SMPL humanoid using NumPy arrays."""

    def __init__(self, source: Path | str = "humenv") -> None:
        super().__init__(source, runtime=NumpyRuntime())


__all__ = ["SmplHumanoid"]
