"""Torch GarmentMeasurements model."""

from pathlib import Path
from typing import Literal

import torch.nn as nn

from body_models.bodies.garment_measurements.model import GarmentMeasurementsModel
from body_models.rotations import RotationType
from body_models.runtime import TorchRuntime


class GarmentMeasurements(GarmentMeasurementsModel, nn.Module):
    """GarmentMeasurements using Torch tensors and optional Warp kernels."""

    skinning_backends = TorchRuntime.skinning_backends

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
