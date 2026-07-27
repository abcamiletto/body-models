"""JAX BrainCo Revo 2 model."""

from pathlib import Path

import jax

from body_models._runtime import JaxModel, JaxRuntime
from body_models.brainco._io import Side
from body_models.brainco._model import BrainCoHandModel


@jax.tree_util.register_pytree_node_class
class BrainCoHand(BrainCoHandModel, JaxModel):
    """BrainCo Revo 2 using JAX arrays."""

    def __init__(self, model_path: Path | str | None = None, *, side: Side = "right") -> None:
        BrainCoHandModel.__init__(
            self,
            model_path,
            side=side,
            runtime=JaxRuntime(),
        )


__all__ = ["BrainCoHand"]
