"""NumPy MHR model."""

from pathlib import Path

from body_models.bodies.mhr.model import MHRModel
from body_models.runtime import NumpyRuntime
from body_models.state import numpy_state


class MHR(MHRModel):
    """MHR using NumPy arrays."""

    skinning_backends = ("numpy",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        lod: int = 1,
        simplify: float = 1.0,
    ) -> None:
        super().__init__(
            model_path,
            lod=lod,
            simplify=simplify,
            runtime=NumpyRuntime(),
            materialize=numpy_state,
        )


__all__ = ["MHR"]
