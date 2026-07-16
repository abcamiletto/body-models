"""JAX SOMA model."""

from pathlib import Path

import jax

from body_models.bodies.soma import correctives_jax
from body_models.bodies.soma.identities import jax as identity_lowerings
from body_models.bodies.soma.io import public_joint_metadata
from body_models.bodies.soma.lowerings import SomaLowerings
from body_models.bodies.soma.model import SOMAModel
from body_models.rotations import RotationType
from body_models.runtime import JaxRuntime
from body_models.state import jax_state

_LOWERINGS = SomaLowerings(correctives_jax.JaxCorrectiveNetwork, identity_lowerings.create_identity_source)


@jax.tree_util.register_pytree_node_class
class SOMA(SOMAModel):
    """SOMA using JAX arrays."""

    skinning_backends = ("jax",)

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

    def tree_flatten(self):
        return (self.weights, self._identity_source), self._config

    @classmethod
    def tree_unflatten(cls, config, children):
        obj = cls.__new__(cls)
        obj._runtime = JaxRuntime()
        obj._config = config
        obj.weights, obj._identity_source = children
        obj.parents, obj._joint_names = public_joint_metadata(obj.weights)
        obj._corrective_network = _LOWERINGS.corrective_network(obj._runtime, obj.weights)
        return obj


__all__ = ["SOMA"]
