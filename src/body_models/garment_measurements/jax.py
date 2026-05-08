"""JAX backend for the GarmentMeasurements PCA body model."""

from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from nanomanifold import SO3

from .. import common
from ..base import BodyModel
from ..rotations import VALID_ROTATION_TYPES, RotationType
from .backends import jax as backend
from .io import get_model_path, load_model_data


__all__ = ["GarmentMeasurements"]


class GarmentMeasurements(BodyModel):
    """GarmentMeasurements PCA body model with FBX-derived skeleton/skinning."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")

        self.weights = common.jaxify(load_model_data(get_model_path(model_path)))
        self.rotation_type = rotation_type

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.joint_names)

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self.weights.mean_vertices.shape[0]

    @property
    def num_shape_components(self) -> int:
        return self.weights.eigenvalues.shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V J"]:
        return self.weights.skin_weights

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.weights.mean_vertices

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        shape: Float[jax.Array, "B C"],
        pose: Float[jax.Array, "B J N"] | Float[jax.Array, "B J 3 3"] | None = None,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        vertex_indices: list[int] | None = None,
    ) -> Float[jax.Array, "B V 3"]:
        return backend.forward_vertices(
            weights=self.weights,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[jax.Array, "B C"],
        pose: Float[jax.Array, "B J N"] | Float[jax.Array, "B J 3 3"] | None = None,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[jax.Array, "B J 4 4"]:
        return backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jax.Array]:
        pose_ref = jnp.zeros((batch_size, self.num_joints, 3), dtype=dtype)
        global_ref = jnp.zeros((batch_size,), dtype=dtype)
        return {
            "shape": jnp.zeros((1, self.num_shape_components), dtype=dtype),
            "pose": SO3.identity_as(
                pose_ref,
                batch_dims=(batch_size, self.num_joints),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_rotation": SO3.identity_as(
                global_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }
