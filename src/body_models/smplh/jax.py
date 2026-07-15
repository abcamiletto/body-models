"""JAX SMPL-H model."""

from pathlib import Path
from typing import Literal

import jax

from body_models.bodies.smplh.model import SMPLHModel
from body_models.rotations import RotationType
from body_models.runtime import JaxModel, JaxRuntime


@jax.tree_util.register_pytree_node_class
class SMPLH(SMPLHModel, JaxModel):
    """SMPL-H using JAX arrays."""

    skinning_backends = ("jax",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        SMPLHModel.__init__(
            self,
            model_path,
            gender,
            flat_hand_mean,
            simplify,
            rotation_type,
            runtime=JaxRuntime(),
        )


__all__ = ["SMPLH"]
