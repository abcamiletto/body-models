"""JAX SMPL humanoid model."""

from pathlib import Path

import jax

from body_models.robots.smpl_humanoid.model import SmplHumanoidModel
from body_models.runtime import JaxModel, JaxRuntime


@jax.tree_util.register_pytree_node_class
class SmplHumanoid(SmplHumanoidModel, JaxModel):
    """SMPL humanoid using JAX arrays."""

    kernels = ("jax",)

    def __init__(self, source: Path | str = "humenv") -> None:
        SmplHumanoidModel.__init__(self, source, runtime=JaxRuntime())


__all__ = ["SmplHumanoid"]
