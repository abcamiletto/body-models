"""NumPy MHR model."""

from pathlib import Path

from body_models._runtime import NumpyRuntime
from body_models.mhr._model import MHRModel


class MHR(MHRModel):
    """MHR using NumPy arrays."""

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
        )


__all__ = ["MHR"]
