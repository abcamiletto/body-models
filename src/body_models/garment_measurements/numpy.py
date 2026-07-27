"""NumPy GarmentMeasurements model."""

from pathlib import Path

from body_models._rotations import RotationType
from body_models._runtime import NumpyRuntime
from body_models.garment_measurements._model import GarmentMeasurementsModel


class GarmentMeasurements(GarmentMeasurementsModel):
    """GarmentMeasurements using NumPy arrays."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        super().__init__(
            model_path,
            rotation_type=rotation_type,
            runtime=NumpyRuntime(),
        )


__all__ = ["GarmentMeasurements"]
