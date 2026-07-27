"""NumPy BrainCo Revo 2 model."""

from pathlib import Path

from body_models._runtime import NumpyRuntime
from body_models.brainco._io import Side
from body_models.brainco._model import BrainCoHandModel


class BrainCoHand(BrainCoHandModel):
    """BrainCo Revo 2 using NumPy arrays."""

    def __init__(self, model_path: Path | str | None = None, *, side: Side = "right") -> None:
        super().__init__(model_path, side=side, runtime=NumpyRuntime())


__all__ = ["BrainCoHand"]
