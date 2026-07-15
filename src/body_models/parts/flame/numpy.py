"""NumPy FLAME model."""

from pathlib import Path

from body_models.parts.flame.model import FLAMEModel
from body_models.rotations import RotationType
from body_models.runtime import NumpyRuntime


class FLAME(FLAMEModel):
    """FLAME using NumPy arrays."""

    kernels = ("numpy",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        super().__init__(model_path, simplify, rotation_type, runtime=NumpyRuntime())


__all__ = ["FLAME"]
