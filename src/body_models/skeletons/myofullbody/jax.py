"""JAX MyoFullBody model."""

from pathlib import Path

import jax

from body_models.runtime import JaxModel, JaxRuntime
from body_models.skeletons.myofullbody.model import MyoFullBodyModel


@jax.tree_util.register_pytree_node_class
class MyoFullBody(MyoFullBodyModel, JaxModel):
    """MyoFullBody using JAX arrays."""

    kernels = ("jax",)

    def __init__(self, model_path: Path | str | None = None) -> None:
        MyoFullBodyModel.__init__(self, model_path, runtime=JaxRuntime())


__all__ = ["MyoFullBody"]
