"""JAX Unitree G1 model."""

from pathlib import Path

import jax

from body_models.robots.g1 import core
from body_models.robots.g1.model import G1Model
from body_models.runtime import JaxModel, JaxRuntime
from body_models.state import jax_state


@jax.tree_util.register_pytree_node_class
class G1(G1Model, JaxModel):
    """Unitree G1 using JAX arrays."""

    skinning_backends = ("jax",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        convention: core.Convention = "soma",
    ) -> None:
        G1Model.__init__(
            self,
            model_path,
            convention=convention,
            runtime=JaxRuntime(),
            materialize=jax_state,
        )


__all__ = ["G1"]
