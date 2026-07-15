"""NumPy SOMA model."""

from pathlib import Path
from typing import Literal

from body_models.bodies.soma.model import SOMAModel
from body_models.rotations import RotationType
from body_models.runtime import NumpyRuntime


class SOMA(SOMAModel):
    """SOMA using NumPy arrays and SciPy sparse correctives."""

    kernels = ("numpy",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        model_type: str = "soma",
        lod: str = "mid",
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        match_warp: bool = True,
        kernel: Literal["numpy"] = "numpy",
    ) -> None:
        if kernel != "numpy":
            raise ValueError(f"Invalid NumPy kernel: {kernel!r}")
        super().__init__(
            model_path,
            model_type=model_type,
            lod=lod,
            simplify=simplify,
            rotation_type=rotation_type,
            match_warp=match_warp,
            runtime=NumpyRuntime(),
        )


__all__ = ["SOMA"]
