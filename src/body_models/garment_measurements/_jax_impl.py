"""JAX backend for the GarmentMeasurements body model."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Float, Int
from nanomanifold import SO3

from ..base import BodyModel
from ..rotations import VALID_ROTATION_TYPES
from . import core
from .io import compute_kinematic_fronts, load_model_data

__all__ = ["GarmentMeasurements"]


class GarmentMeasurements(BodyModel, nnx.Module):
    """GarmentMeasurements PCA body model with FBX-derived skeleton/skinning."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: core.RotationType = "axis_angle",
    ) -> None:
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")

        data = load_model_data(model_path=model_path, dtype=np.float32)
        self.mean_vertices = nnx.Variable(jnp.asarray(data["mean_vertices"]))
        self.components = nnx.Variable(jnp.asarray(data["components"]))
        self.eigenvalues = nnx.Variable(jnp.asarray(data["eigenvalues"]))
        self.bind_quats = nnx.Variable(jnp.asarray(data["bind_quats"]))
        self._skin_weights = nnx.Variable(jnp.asarray(data["skin_weights"]))
        self.mvc_weights = nnx.Variable(jnp.asarray(data["mvc_weights"]))
        self._faces = nnx.Variable(jnp.asarray(data["faces"]))
        self.parents = data["parents"].astype(int).tolist()
        self._kinematic_fronts = compute_kinematic_fronts(self.parents)
        self._joint_names = list(data["joint_names"])
        self.rotation_type = rotation_type

    @property
    def faces(self) -> Int[jax.Array, "F _"]:
        return self._faces[...]

    @property
    def num_joints(self) -> int:
        return len(self._joint_names)

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return self.mean_vertices[...].shape[0]

    @property
    def num_shape_components(self) -> int:
        return self.eigenvalues[...].shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V J"]:
        return self._skin_weights[...]

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.mean_vertices[...]

    def forward_vertices(
        self,
        shape: Float[jax.Array, "B C"],
        pose: Float[jax.Array, "B J N"] | Float[jax.Array, "B J 3 3"] | None = None,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        vertex_indices: list[int] | None = None,
    ) -> Float[jax.Array, "B V 3"]:
        return core.forward_vertices(
            mean_vertices=self.mean_vertices[...],
            components=self.components[...],
            eigenvalues=self.eigenvalues[...],
            bind_quats=self.bind_quats[...],
            skin_weights=self._skin_weights[...],
            mvc_weights=self.mvc_weights[...],
            kinematic_fronts=self._kinematic_fronts,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
            xp=jnp,
        )

    def forward_skeleton(
        self,
        shape: Float[jax.Array, "B C"],
        pose: Float[jax.Array, "B J N"] | Float[jax.Array, "B J 3 3"] | None = None,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[jax.Array, "B J 4 4"]:
        return core.forward_skeleton(
            mean_vertices=self.mean_vertices[...],
            components=self.components[...],
            eigenvalues=self.eigenvalues[...],
            bind_quats=self.bind_quats[...],
            mvc_weights=self.mvc_weights[...],
            kinematic_fronts=self._kinematic_fronts,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
            xp=jnp,
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
