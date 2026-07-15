"""JAX SMPL model."""

from pathlib import Path
from typing import Literal

import jax

from body_models.bodies.smpl.model import SMPLModel
from body_models.rotations import RotationType
from body_models.runtime import JaxModel, JaxRuntime


@jax.tree_util.register_pytree_node_class
class SMPL(SMPLModel, JaxModel):
    """SMPL using JAX arrays."""

    kernels = ("jax",)

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        SMPLModel.__init__(
            self,
            model_path,
            gender,
            simplify,
            rotation_type,
            runtime=JaxRuntime(),
        )


__all__ = ["SMPL"]
