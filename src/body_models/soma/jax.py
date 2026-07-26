"""JAX SOMA model."""

from pathlib import Path

import jax

from body_models.rotations import RotationType
from body_models.runtime import JaxModel, JaxRuntime
from body_models.soma import correctives_jax
from body_models.soma.io import public_joint_metadata
from body_models.soma.model import SOMAModel


@jax.tree_util.register_pytree_node_class
class SOMA(SOMAModel, JaxModel):
    """SOMA using JAX arrays."""

    _jax_children = ("weights", "_identity_source")

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        model_type: str = "soma",
        lod: str = "mid",
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        match_warp: bool = True,
    ) -> None:
        super().__init__(
            model_path,
            model_type=model_type,
            lod=lod,
            simplify=simplify,
            rotation_type=rotation_type,
            match_warp=match_warp,
            runtime=JaxRuntime(),
            corrective_network=correctives_jax.JaxCorrectiveNetwork,
        )

    def _rebuild_jax_state(self) -> None:
        self.parents, self._joint_names = public_joint_metadata(self.weights)
        self._corrective_network = correctives_jax.JaxCorrectiveNetwork(self._runtime, self.weights)


__all__ = ["SOMA"]
