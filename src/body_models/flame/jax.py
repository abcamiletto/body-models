"""JAX backend for FLAME model."""

from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from body_models import common
from body_models.base import BodyModel
from nanomanifold import SO3

from body_models.flame.backends import jax as backend
from body_models.flame.constants import FLAME_JOINT_NAMES
from body_models.flame.io import get_model_path, load_model_data
from body_models.rotations import VALID_ROTATION_TYPES, RotationType

__all__ = ["FLAME"]


class FLAME(BodyModel):
    """FLAME head model with JAX backend."""

    NUM_HEAD_JOINTS = 4
    NUM_JOINTS = 5

    def __init__(
        self,
        model_path: Path | str | None = None,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ):
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1

        resolved_path = get_model_path(model_path)
        weights = load_model_data(resolved_path, simplify=simplify)
        self.weights = common.jaxify(weights)

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return list(FLAME_JOINT_NAMES)

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V 5"]:
        return self.weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.weights.v_template

    @property
    def shapedirs(self) -> Float[jax.Array, "V 3 S"]:
        return self.weights.shapedirs

    @property
    def exprdirs(self) -> Float[jax.Array, "V 3 E"]:
        return self.weights.exprdirs

    @property
    def posedirs(self) -> Float[jax.Array, "P V*3"]:
        return self.weights.posedirs

    @property
    def lbs_weights(self) -> Float[jax.Array, "V 5"]:
        return self.weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        shape: Float[jax.Array, "B|1 S"],
        expression: Float[jax.Array, "B E"],
        head_pose: Float[jax.Array, "B 4 N"] | Float[jax.Array, "B 4 3 3"],
        head_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[jax.Array, "B V 3"]:
        return backend.forward_vertices(
            weights=self.weights,
            shape=shape,
            expression=expression,
            pose=head_pose,
            head_rotation=head_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[jax.Array, "B|1 S"],
        expression: Float[jax.Array, "B E"],
        head_pose: Float[jax.Array, "B 4 N"] | Float[jax.Array, "B 4 3 3"],
        head_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[jax.Array, "B 5 4 4"]:
        return backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            expression=expression,
            pose=head_pose,
            head_rotation=head_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jax.Array]:
        ref = jnp.zeros((batch_size, 100), dtype=dtype)
        return {
            "shape": jnp.zeros((1, 300), dtype=dtype),
            "expression": jnp.zeros((batch_size, 100), dtype=dtype),
            "head_pose": SO3.identity_as(
                ref,
                batch_dims=(batch_size, self.NUM_HEAD_JOINTS),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "head_rotation": SO3.identity_as(ref, batch_dims=(batch_size,), rotation_type=self.rotation_type, xp=jnp),
            "global_rotation": SO3.identity_as(ref, batch_dims=(batch_size,), rotation_type=self.rotation_type, xp=jnp),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }
