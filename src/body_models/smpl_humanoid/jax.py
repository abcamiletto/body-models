"""JAX SMPL humanoid model."""

from pathlib import Path

import jax

from body_models._runtime import JaxModel, JaxRuntime
from body_models.smpl_humanoid._model import SmplHumanoidModel


@jax.tree_util.register_pytree_node_class
class SmplHumanoid(SmplHumanoidModel, JaxModel):
    """SMPL humanoid using JAX arrays."""

    def __init__(self, source: Path | str = "humenv") -> None:
        SmplHumanoidModel.__init__(
            self,
            source,
            runtime=JaxRuntime(),
        )


__all__ = ["SmplHumanoid"]
