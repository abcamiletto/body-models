"""JAX backend kernels for BrainCo."""

import jax
import jax.numpy as jnp
from jaxtyping import Float

from body_models.robots.brainco.backends import core
from body_models.robots.brainco.io import BrainCoWeights


def forward_skeleton(
    weights: BrainCoWeights,
    pose: Float[jax.Array, "B Q N"] | Float[jax.Array, "B Q 3 3"],
    global_translation: Float[jax.Array, "B 3"] | None = None,
    *,
    global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    joint_indices=None,
    rotation_type: core.RotationType = "rotmat",
):
    return core.forward_skeleton(
        local_offsets=weights.local_offsets,
        rest_local_rotations=weights.rest_local_rotations,
        joint_axes=weights.qpos_joint_axes,
        joint_indices=weights.qpos_joint_indices,
        coupled_joint_axes=weights.coupled_joint_axes,
        coupled_joint_indices=weights.coupled_joint_indices,
        coupled_driver_indices=weights.coupled_driver_indices,
        coupled_polycoef=weights.coupled_polycoef,
        parents=weights.parents,
        pose=pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        skeleton_indices=joint_indices,
        rotation_type=rotation_type,
        xp=jnp,
    )


def forward_links(
    weights: BrainCoWeights,
    pose: Float[jax.Array, "B Q N"] | Float[jax.Array, "B Q 3 3"],
    global_translation: Float[jax.Array, "B 3"] | None = None,
    *,
    global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    rotation_type: core.RotationType = "rotmat",
):
    return core.forward_links(
        local_offsets=weights.local_offsets,
        rest_local_rotations=weights.rest_local_rotations,
        joint_axes=weights.qpos_joint_axes,
        joint_indices=weights.qpos_joint_indices,
        coupled_joint_axes=weights.coupled_joint_axes,
        coupled_joint_indices=weights.coupled_joint_indices,
        coupled_driver_indices=weights.coupled_driver_indices,
        coupled_polycoef=weights.coupled_polycoef,
        parents=weights.parents,
        link_joint_indices=weights.link_joint_indices,
        link_geom_positions=weights.link_geom_positions,
        link_geom_rotations=weights.link_geom_rotations,
        pose=pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        rotation_type=rotation_type,
        xp=jnp,
    )


def forward_meshes(
    weights: BrainCoWeights,
    pose: Float[jax.Array, "B Q N"] | Float[jax.Array, "B Q 3 3"],
    global_translation: Float[jax.Array, "B 3"] | None = None,
    *,
    global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    rotation_type: core.RotationType = "rotmat",
):
    return core.forward_meshes(
        vertices=weights.vertices,
        faces=weights.faces,
        local_offsets=weights.local_offsets,
        rest_local_rotations=weights.rest_local_rotations,
        joint_axes=weights.qpos_joint_axes,
        joint_indices=weights.qpos_joint_indices,
        coupled_joint_axes=weights.coupled_joint_axes,
        coupled_joint_indices=weights.coupled_joint_indices,
        coupled_driver_indices=weights.coupled_driver_indices,
        coupled_polycoef=weights.coupled_polycoef,
        parents=weights.parents,
        link_joint_indices=weights.link_joint_indices,
        link_vertex_starts=weights.link_vertex_starts,
        link_vertex_counts=weights.link_vertex_counts,
        link_face_starts=weights.link_face_starts,
        link_face_counts=weights.link_face_counts,
        link_geom_positions=weights.link_geom_positions,
        link_geom_rotations=weights.link_geom_rotations,
        pose=pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        rotation_type=rotation_type,
        xp=jnp,
    )
