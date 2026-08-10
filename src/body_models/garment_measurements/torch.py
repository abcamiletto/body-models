"""PyTorch GarmentMeasurements model."""

from body_models._backend import model_for_backend
from body_models.garment_measurements._model import GarmentMeasurements as _GarmentMeasurements

GarmentMeasurements = model_for_backend(_GarmentMeasurements, "torch", module=__name__)

__all__ = ["GarmentMeasurements"]
