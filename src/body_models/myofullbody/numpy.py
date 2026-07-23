"""NumPy MyoFullBody model."""

from pathlib import Path

from body_models.runtime import NumpyRuntime
from body_models.state import numpy_state
from body_models.skeletons.myofullbody.model import MyoFullBodyModel


class MyoFullBody(MyoFullBodyModel):
    """MyoFullBody using NumPy arrays."""

    skinning_backends = ("numpy",)

    def __init__(self, model_path: Path | str | None = None) -> None:
        super().__init__(model_path, runtime=NumpyRuntime(), materialize=numpy_state)


__all__ = ["MyoFullBody"]
