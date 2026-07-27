"""NumPy SKEL model."""

from pathlib import Path
from typing import Literal

from body_models._runtime import NumpyRuntime
from body_models.skel._model import SKELModel


class SKEL(SKELModel):
    """SKEL using NumPy arrays."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["male", "female"] | None = None,
        simplify: float = 1.0,
    ) -> None:
        super().__init__(
            model_path,
            gender,
            simplify,
            runtime=NumpyRuntime(),
        )


__all__ = ["SKEL"]
