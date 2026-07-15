"""NumPy SMPL model."""

from pathlib import Path
from typing import Literal

from body_models.bodies.smpl.model import SMPLModel
from body_models.rotations import RotationType
from body_models.runtime import NumpyRuntime


class SMPL(SMPLModel):
    """SMPL using NumPy arrays."""

    skinning_backends = ("numpy",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        super().__init__(
            model_path,
            gender,
            simplify,
            rotation_type,
            runtime=NumpyRuntime(),
        )


__all__ = ["SMPL"]
