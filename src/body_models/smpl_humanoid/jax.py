"""JAX SMPL humanoid model."""

from pathlib import Path

import jax

from body_models.robots.smpl_humanoid.model import SmplHumanoidModel
from body_models.runtime import JaxModel, JaxRuntime
from body_models.state import jax_state


@jax.tree_util.register_pytree_node_class
class SmplHumanoid(SmplHumanoidModel, JaxModel):
    """SMPL humanoid using JAX arrays."""

    skinning_backends = ("jax",)

    def __init__(self, source: Path | str = "humenv") -> None:
        SmplHumanoidModel.__init__(
            self,
            source,
            runtime=JaxRuntime(),
            materialize=jax_state,
        )


__all__ = ["SmplHumanoid"]
