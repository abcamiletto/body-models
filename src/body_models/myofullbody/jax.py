"""JAX MyoFullBody model."""

from pathlib import Path

import jax

from body_models.runtime import JaxModel, JaxRuntime
from body_models.state import jax_state
from body_models.skeletons.myofullbody.model import MyoFullBodyModel


@jax.tree_util.register_pytree_node_class
class MyoFullBody(MyoFullBodyModel, JaxModel):
    """MyoFullBody using JAX arrays."""

    skinning_backends = ("jax",)

    def __init__(self, model_path: Path | str | None = None) -> None:
        MyoFullBodyModel.__init__(
            self,
            model_path,
            runtime=JaxRuntime(),
            materialize=jax_state,
        )


__all__ = ["MyoFullBody"]
