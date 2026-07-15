"""JAX SKEL model."""

from pathlib import Path
from typing import Literal

import jax

from body_models.runtime import JaxModel, JaxRuntime
from body_models.skeletons.skel.model import SKELModel


@jax.tree_util.register_pytree_node_class
class SKEL(SKELModel, JaxModel):
    """SKEL using JAX arrays."""

    kernels = ("jax",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["male", "female"] | None = None,
        simplify: float = 1.0,
    ) -> None:
        SKELModel.__init__(self, model_path, gender, simplify, runtime=JaxRuntime())


__all__ = ["SKEL"]
