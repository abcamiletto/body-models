"""JAX MHR model."""

from pathlib import Path

import jax

from body_models.bodies.mhr.model import MHRModel
from body_models.runtime import JaxModel, JaxRuntime


@jax.tree_util.register_pytree_node_class
class MHR(MHRModel, JaxModel):
    """MHR using JAX arrays."""

    skinning_backends = ("jax",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        lod: int = 1,
        simplify: float = 1.0,
    ) -> None:
        MHRModel.__init__(
            self,
            model_path,
            lod=lod,
            simplify=simplify,
            runtime=JaxRuntime(),
        )


__all__ = ["MHR"]
