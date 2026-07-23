"""JAX GarmentMeasurements model."""

from pathlib import Path

import jax

from body_models.bodies.garment_measurements.model import GarmentMeasurementsModel
from body_models.rotations import RotationType
from body_models.runtime import JaxModel, JaxRuntime
from body_models.state import jax_state


@jax.tree_util.register_pytree_node_class
class GarmentMeasurements(GarmentMeasurementsModel, JaxModel):
    """GarmentMeasurements using JAX arrays."""

    skinning_backends = ("jax",)

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
            materialize=jax_state,
        )


__all__ = ["GarmentMeasurements"]
