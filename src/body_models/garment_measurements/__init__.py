"""Public GarmentMeasurements API."""

from body_models.garment_measurements._model import (
    GarmentMeasurements,
    GarmentMeasurementsIdentity,
    GarmentMeasurementsPreparedPose,
)

GarmentMeasurements.__module__ = __name__

__all__ = [
    "GarmentMeasurements",
    "GarmentMeasurementsIdentity",
    "GarmentMeasurementsPreparedPose",
]
