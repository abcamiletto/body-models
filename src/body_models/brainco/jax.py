"""JAX BrainCo Revo 2 model."""

from pathlib import Path

import jax

from body_models.robots.brainco.io import Side
from body_models.robots.brainco.model import BrainCoHandModel
from body_models.runtime import JaxModel, JaxRuntime


@jax.tree_util.register_pytree_node_class
class BrainCoHand(BrainCoHandModel, JaxModel):
    """BrainCo Revo 2 using JAX arrays."""

    skinning_backends = ("jax",)

    def __init__(self, model_path: Path | str | None = None, *, side: Side = "right") -> None:
        BrainCoHandModel.__init__(self, model_path, side=side, runtime=JaxRuntime())


__all__ = ["BrainCoHand"]
