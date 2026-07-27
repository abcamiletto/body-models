"""JAX GarmentMeasurements model."""

from pathlib import Path

import jax

from body_models._rotations import RotationType
from body_models._runtime import JaxModel, JaxRuntime
from body_models.garment_measurements._model import GarmentMeasurementsModel


@jax.tree_util.register_pytree_node_class
class GarmentMeasurements(GarmentMeasurementsModel, JaxModel):
    """GarmentMeasurements using JAX arrays."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        GarmentMeasurementsModel.__init__(
            self,
            model_path,
            rotation_type=rotation_type,
            runtime=JaxRuntime(),
        )


__all__ = ["GarmentMeasurements"]
