"""NumPy Unitree G1 model."""

from pathlib import Path

from body_models.g1 import core
from body_models.g1.model import G1Model
from body_models.runtime import NumpyRuntime


class G1(G1Model):
    """Unitree G1 using NumPy arrays."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        convention: core.Convention = "soma",
    ) -> None:
        super().__init__(
            model_path,
            convention=convention,
            runtime=NumpyRuntime(),
        )


__all__ = ["G1"]
