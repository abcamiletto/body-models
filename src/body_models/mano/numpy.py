"""NumPy MANO model."""

from pathlib import Path
from typing import Literal

from body_models.parts.mano.model import MANOModel
from body_models.rotations import RotationType
from body_models.runtime import NumpyRuntime


class MANO(MANOModel):
    """MANO using NumPy arrays."""

    skinning_backends = ("numpy",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        side: Literal["right", "left"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        super().__init__(
            model_path,
            side,
            flat_hand_mean,
            simplify,
            rotation_type,
            runtime=NumpyRuntime(),
        )


__all__ = ["MANO"]
