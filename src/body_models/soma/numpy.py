"""NumPy SOMA model."""

from pathlib import Path

from body_models.soma import correctives_numpy
from body_models.soma.model import SOMAModel
from body_models.rotations import RotationType
from body_models.runtime import NumpyRuntime

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
        match_warp: bool = True,
    ) -> None:
        super().__init__(
            model_path,
            model_type=model_type,
            lod=lod,
            simplify=simplify,
            rotation_type=rotation_type,
            match_warp=match_warp,
            runtime=NumpyRuntime(),
            corrective_network=correctives_numpy.NumpyCorrectiveNetwork,
        )


__all__ = ["SOMA"]
