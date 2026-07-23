"""NumPy BrainCo Revo 2 model."""

from pathlib import Path

from body_models.robots.brainco.io import Side
from body_models.robots.brainco.model import BrainCoHandModel
from body_models.runtime import NumpyRuntime
from body_models.state import numpy_state


class BrainCoHand(BrainCoHandModel):
    """BrainCo Revo 2 using NumPy arrays."""

    skinning_backends = ("numpy",)

    def __init__(self, model_path: Path | str | None = None, *, side: Side = "right") -> None:
        super().__init__(model_path, side=side, runtime=NumpyRuntime(), materialize=numpy_state)


__all__ = ["BrainCoHand"]
