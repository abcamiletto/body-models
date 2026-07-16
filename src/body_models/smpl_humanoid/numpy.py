"""NumPy SMPL humanoid model."""

from pathlib import Path

from body_models.robots.smpl_humanoid.model import SmplHumanoidModel
from body_models.runtime import NumpyRuntime
from body_models.state import numpy_state


class SmplHumanoid(SmplHumanoidModel):
    """SMPL humanoid using NumPy arrays."""

    skinning_backends = ("numpy",)

    def __init__(self, source: Path | str = "humenv") -> None:
        super().__init__(source, runtime=NumpyRuntime(), materialize=numpy_state)


__all__ = ["SmplHumanoid"]
