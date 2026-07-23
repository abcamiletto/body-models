"""JAX ANNY model."""

from pathlib import Path

import jax

from body_models.bodies.anny.model import ANNYModel
from body_models.rotations import RotationType
from body_models.runtime import JaxModel, JaxRuntime
from body_models.state import jax_state


@jax.tree_util.register_pytree_node_class
class ANNY(ANNYModel, JaxModel):
    """ANNY using JAX arrays."""

    skinning_backends = ("jax",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rig: str = "default",
        topology: str = "default",
        all_phenotypes: bool = False,
        extrapolate_phenotypes: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        ANNYModel.__init__(
            self,
            model_path,
            rig=rig,
            topology=topology,
            all_phenotypes=all_phenotypes,
            extrapolate_phenotypes=extrapolate_phenotypes,
            simplify=simplify,
            rotation_type=rotation_type,
            runtime=JaxRuntime(),
            materialize=jax_state,
        )


__all__ = ["ANNY"]
