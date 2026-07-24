"""JAX SOMA model."""

import functools
from pathlib import Path

import jax

from body_models import registry
from body_models.bodies.soma import correctives_jax
from body_models.bodies.soma.identities import jax as identity_lowerings
from body_models.bodies.soma.io import public_joint_metadata
from body_models.bodies.soma.lowerings import SomaLowerings
from body_models.bodies.soma.model import SOMAModel
from body_models.rotations import RotationType
from body_models.runtime import JaxModel, JaxRuntime
from body_models.state import jax_state

_IDENTITY_SOURCE = functools.partial(
    identity_lowerings.create_identity_source,
    model_factory=functools.partial(registry.create_model, backend="jax"),
)
_LOWERINGS = SomaLowerings(correctives_jax.JaxCorrectiveNetwork, _IDENTITY_SOURCE)


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
            materialize=jax_state,
            lowerings=_LOWERINGS,
        )

    def _rebuild_jax_state(self) -> None:
        self.parents, self._joint_names = public_joint_metadata(self.weights)
        self._corrective_network = _LOWERINGS.corrective_network(self._runtime, self.weights)


__all__ = ["SOMA"]
