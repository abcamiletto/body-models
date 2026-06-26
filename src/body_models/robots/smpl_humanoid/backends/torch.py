"""PyTorch backend kernels for SmplHumanoid."""

import torch
from jaxtyping import Float
from trimesh import Trimesh
from torch import Tensor

from body_models.robots.smpl_humanoid.backends import core
from body_models.robots.smpl_humanoid.io import SmplHumanoidWeights


def forward_skeleton(
    weights: SmplHumanoidWeights,
    body_pose: Float[Tensor, "B Q"],
    global_translation: Float[Tensor, "B 3"] | None = None,
    *,
    global_rotation: Float[Tensor, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
) -> Float[Tensor, "B J 4 4"]:
    return core.forward_skeleton(
        local_offsets=weights.local_offsets,
        rest_local_rotations=weights.rest_local_rotations,
        actuated_joint_indices=weights.actuated_joint_indices,
        parents=weights.parents,
        body_pose=body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        joint_indices=joint_indices,
        xp=torch,
    )


def forward_links(
    weights: SmplHumanoidWeights,
    body_pose: Float[Tensor, "B Q"],
    global_translation: Float[Tensor, "B 3"] | None = None,
    *,
    global_rotation: Float[Tensor, "B 3"] | None = None,
) -> Float[Tensor, "B L 4 4"]:
    return core.forward_links(
        local_offsets=weights.local_offsets,
        rest_local_rotations=weights.rest_local_rotations,
        actuated_joint_indices=weights.actuated_joint_indices,
        parents=weights.parents,
        link_joint_indices=weights.link_joint_indices,
        link_geom_positions=weights.link_geom_positions,
        link_geom_rotations=weights.link_geom_rotations,
        body_pose=body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        xp=torch,
    )


def forward_meshes(
    weights: SmplHumanoidWeights,
    body_pose: Float[Tensor, "B Q"],
    global_translation: Float[Tensor, "B 3"] | None = None,
    *,
    global_rotation: Float[Tensor, "B 3"] | None = None,
) -> list[Trimesh]:
    return core.forward_meshes(
        vertices=weights.vertices,
        faces=weights.faces,
        local_offsets=weights.local_offsets,
        rest_local_rotations=weights.rest_local_rotations,
        actuated_joint_indices=weights.actuated_joint_indices,
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
        xp=torch,
    )
