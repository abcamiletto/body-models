"""JAX FLAME model."""

from pathlib import Path

import jax

from body_models._rotations import RotationType
from body_models._runtime import JaxModel, JaxRuntime
from body_models.flame._model import FLAMEModel


@jax.tree_util.register_pytree_node_class
class FLAME(FLAMEModel, JaxModel):
    """FLAME using JAX arrays."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        FLAMEModel.__init__(
            self,
            model_path,
            simplify,
            rotation_type,
            runtime=JaxRuntime(),
        )


__all__ = ["FLAME"]
