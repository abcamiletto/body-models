"""JAX MANO model."""

from pathlib import Path
from typing import Literal

import jax

from body_models.parts.mano.model import MANOModel
from body_models.rotations import RotationType
from body_models.runtime import JaxModel, JaxRuntime
from body_models.state import jax_state


@jax.tree_util.register_pytree_node_class
class MANO(MANOModel, JaxModel):
    """MANO using JAX arrays."""

    skinning_backends = ("jax",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        side: Literal["right", "left"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        MANOModel.__init__(
            self,
            model_path,
            side,
            flat_hand_mean,
            simplify,
            rotation_type,
            runtime=JaxRuntime(),
            materialize=jax_state,
        )


__all__ = ["MANO"]
