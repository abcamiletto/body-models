"""NumPy ANNY model."""

from pathlib import Path

from body_models.bodies.anny.model import ANNYModel
from body_models.rotations import RotationType
from body_models.runtime import NumpyRuntime


class ANNY(ANNYModel):
    """ANNY using NumPy arrays."""

    skinning_backends = ("numpy",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rig: str = "default",
        topology: str = "default",
        all_phenotypes: bool = False,
        extrapolate_phenotypes: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        super().__init__(
            model_path,
            rig=rig,
            topology=topology,
            all_phenotypes=all_phenotypes,
            extrapolate_phenotypes=extrapolate_phenotypes,
            simplify=simplify,
            rotation_type=rotation_type,
            runtime=NumpyRuntime(),
        )


__all__ = ["ANNY"]
