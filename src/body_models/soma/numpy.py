"""NumPy SOMA model."""

from pathlib import Path

from body_models._rotations import RotationType
from body_models._runtime import NumpyRuntime
from body_models.soma._model import SOMAModel


class SOMA(SOMAModel):
    """SOMA using NumPy arrays and SciPy sparse correctives."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        model_type: str = "soma",
        lod: str = "mid",
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        super().__init__(
            model_path,
            model_type=model_type,
            lod=lod,
            simplify=simplify,
            rotation_type=rotation_type,
            runtime=NumpyRuntime(),
        )


__all__ = ["SOMA"]
