"""NumPy SMPL-H model."""

from pathlib import Path
from typing import Literal

from body_models.bodies.smplh.model import SMPLHModel
from body_models.rotations import RotationType
from body_models.runtime import NumpyRuntime


class SMPLH(SMPLHModel):
    """SMPL-H using NumPy arrays."""

    kernels = ("numpy",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        super().__init__(
            model_path,
            gender,
            flat_hand_mean,
            simplify,
            rotation_type,
            runtime=NumpyRuntime(),
        )


__all__ = ["SMPLH"]
