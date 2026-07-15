"""NumPy GarmentMeasurements model."""

from pathlib import Path

from body_models.bodies.garment_measurements.model import GarmentMeasurementsModel
from body_models.rotations import RotationType
from body_models.runtime import NumpyRuntime


class GarmentMeasurements(GarmentMeasurementsModel):
    """GarmentMeasurements using NumPy arrays."""

    skinning_backends = ("numpy",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        super().__init__(model_path, rotation_type=rotation_type, runtime=NumpyRuntime())


__all__ = ["GarmentMeasurements"]
