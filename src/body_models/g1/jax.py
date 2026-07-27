"""JAX Unitree G1 model."""

from pathlib import Path

import jax

from body_models._runtime import JaxModel, JaxRuntime
from body_models.g1 import _core as core
from body_models.g1._model import G1Model


@jax.tree_util.register_pytree_node_class
class G1(G1Model, JaxModel):
    """Unitree G1 using JAX arrays."""

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
        )


__all__ = ["G1"]
