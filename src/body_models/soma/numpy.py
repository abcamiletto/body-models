"""NumPy SOMA model."""

from pathlib import Path

from body_models.bodies.soma import correctives_numpy
from body_models.bodies.soma.identities import numpy as identity_lowerings
from body_models.bodies.soma.lowerings import SomaLowerings
from body_models.bodies.soma.model import SOMAModel
from body_models.rotations import RotationType
from body_models.runtime import NumpyRuntime
from body_models.state import numpy_state

_LOWERINGS = SomaLowerings(correctives_numpy.NumpyCorrectiveNetwork, identity_lowerings.create_identity_source)


class SOMA(SOMAModel):
    """SOMA using NumPy arrays and SciPy sparse correctives."""

    skinning_backends = ("numpy",)

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
            materialize=numpy_state,
            lowerings=_LOWERINGS,
        )


__all__ = ["SOMA"]
