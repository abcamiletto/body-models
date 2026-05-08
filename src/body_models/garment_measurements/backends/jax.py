"""JAX GarmentMeasurements backend."""

import jax
import jax.numpy as jnp
from jaxtyping import Float

from body_models.garment_measurements.backends.core import forward_skeleton as _forward_skeleton
from body_models.garment_measurements.backends.core import forward_vertices as _forward_vertices
from body_models.garment_measurements.io import GarmentMeasurementsWeights
from body_models.rotations import RotationType

__all__ = ["forward_vertices", "forward_skeleton"]


def forward_vertices(
    weights: GarmentMeasurementsWeights,
    shape: Float[jax.Array, "B C"],
    pose: Float[jax.Array, "B J N"] | Float[jax.Array, "B J 3 3"] | None = None,
    global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    global_translation: Float[jax.Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return _forward_vertices(
        mean_vertices=weights.mean_vertices,
        components=weights.components,
        eigenvalues=weights.eigenvalues,
        bind_quats=weights.bind_quats,
        skin_weights=weights.skin_weights,
        mvc_weights=weights.mvc_weights,
        kinematic_fronts=weights.kinematic_fronts,
        shape=shape,
        pose=pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        xp=jnp,
    )


def forward_skeleton(
    weights: GarmentMeasurementsWeights,
    shape: Float[jax.Array, "B C"],
    pose: Float[jax.Array, "B J N"] | Float[jax.Array, "B J 3 3"] | None = None,
    global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    global_translation: Float[jax.Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return _forward_skeleton(
        mean_vertices=weights.mean_vertices,
        components=weights.components,
        eigenvalues=weights.eigenvalues,
        bind_quats=weights.bind_quats,
        mvc_weights=weights.mvc_weights,
        kinematic_fronts=weights.kinematic_fronts,
        shape=shape,
        pose=pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        xp=jnp,
    )
