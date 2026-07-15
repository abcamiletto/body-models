"""JAX SOMA model."""

from pathlib import Path

from body_models.bodies.soma.model import SOMAModel
from body_models.rotations import RotationType
from body_models.runtime import JaxRuntime


class SOMA(SOMAModel):
    """SOMA using JAX arrays."""

    kernels = ("jax",)

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
            runtime=JaxRuntime(),
        )


__all__ = ["SOMA"]
