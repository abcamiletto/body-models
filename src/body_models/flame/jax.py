"""JAX FLAME model."""

from pathlib import Path

import jax

from body_models.parts.flame.model import FLAMEModel
from body_models.rotations import RotationType
from body_models.runtime import JaxModel, JaxRuntime
from body_models.state import jax_state


@jax.tree_util.register_pytree_node_class
class FLAME(FLAMEModel, JaxModel):
    """FLAME using JAX arrays."""

    skinning_backends = ("jax",)

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
            materialize=jax_state,
        )


__all__ = ["FLAME"]
