"""NumPy backend kernels for G1."""

import numpy as np
from jaxtyping import Float

from body_models.robots.g1.backends import core
from body_models.robots.g1.io import G1Weights


def forward_skeleton(
    weights: G1Weights,
    body_pose: Float[np.ndarray, "B Q"],
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    *,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: core.RotationType = "rotmat",
):
    return core.forward_skeleton(
        local_offsets=weights.local_offsets,
        rest_local_rotations=weights.rest_local_rotations,
        actuated_joint_indices=weights.actuated_joint_indices,
        actuated_joint_axes=weights.actuated_joint_axes,
        parents=weights.parents,
        body_pose=body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        xp=np,
    )


def forward_links(
    weights: G1Weights,
    body_pose: Float[np.ndarray, "B Q"],
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    *,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    rotation_type: core.RotationType = "rotmat",
):
    return core.forward_links(
        local_offsets=weights.local_offsets,
        rest_local_rotations=weights.rest_local_rotations,
        actuated_joint_indices=weights.actuated_joint_indices,
        actuated_joint_axes=weights.actuated_joint_axes,
        parents=weights.parents,
        link_joint_indices=weights.link_joint_indices,
        link_geom_positions=weights.link_geom_positions,
        link_geom_rotations=weights.link_geom_rotations,
        body_pose=body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        rotation_type=rotation_type,
        xp=np,
    )


def forward_meshes(
    weights: G1Weights,
    body_pose: Float[np.ndarray, "B Q"],
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    *,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    rotation_type: core.RotationType = "rotmat",
):
    return core.forward_meshes(
        vertices=weights.vertices,
        faces=weights.faces,
        local_offsets=weights.local_offsets,
        rest_local_rotations=weights.rest_local_rotations,
        actuated_joint_indices=weights.actuated_joint_indices,
        actuated_joint_axes=weights.actuated_joint_axes,
        parents=weights.parents,
        link_joint_indices=weights.link_joint_indices,
        link_vertex_starts=weights.link_vertex_starts,
        link_vertex_counts=weights.link_vertex_counts,
        link_face_starts=weights.link_face_starts,
        link_face_counts=weights.link_face_counts,
        link_geom_positions=weights.link_geom_positions,
        link_geom_rotations=weights.link_geom_rotations,
        body_pose=body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        rotation_type=rotation_type,
        xp=np,
    )
