"""JAX SMPL-X model."""

from pathlib import Path
from typing import Literal

import jax

from body_models.bodies.smplx.model import SMPLXModel
from body_models.rotations import RotationType
from body_models.runtime import JaxModel, JaxRuntime


@jax.tree_util.register_pytree_node_class
class SMPLX(SMPLXModel, JaxModel):
    """SMPL-X using JAX arrays."""

    kernels = ("jax",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        SMPLXModel.__init__(
            self,
            model_path,
            gender,
            flat_hand_mean,
            simplify,
            rotation_type,
            runtime=JaxRuntime(),
        )


__all__ = ["SMPLX"]
