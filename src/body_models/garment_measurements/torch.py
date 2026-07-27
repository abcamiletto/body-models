"""Torch GarmentMeasurements model."""

from pathlib import Path
from typing import Literal

from torch import nn

from body_models._rotations import RotationType
from body_models._runtime import TorchRuntime
from body_models.garment_measurements._model import GarmentMeasurementsModel


class GarmentMeasurements(GarmentMeasurementsModel, nn.Module):
    """GarmentMeasurements using Torch tensors and optional Warp kernels."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: RotationType = "axis_angle",
        skinning_backend: Literal["torch", "warp"] = "torch",
    ) -> None:
        nn.Module.__init__(self)
        GarmentMeasurementsModel.__init__(
            self,
            model_path,
            rotation_type=rotation_type,
            runtime=TorchRuntime(skinning_backend),
        )


__all__ = ["GarmentMeasurements"]
