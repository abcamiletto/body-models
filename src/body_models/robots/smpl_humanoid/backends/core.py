"""Backend-agnostic SMPL humanoid rigid articulated model computation."""

from __future__ import annotations

from typing import Any

from array_api_compat import get_namespace
from jaxtyping import Float, Int
from nanomanifold import SO3
from trimesh import Trimesh

from body_models.common import rigid
from body_models.rotations import RotationType as SO3RotationType

Array = Any
RotationType = SO3RotationType

VALID_ROTATION_TYPES = ("axis_angle", "quat", "sixd", "matrix", "rotmat")


def _body_rotations(
    body_pose: Float[Array, "B Q"],
    num_actuated_joints: int,
    *,
    xp: Any,
) -> Float[Array, "B A 3 3"]:
    expected = 3 * num_actuated_joints
    if body_pose.ndim < 1 or body_pose.shape[-1] != expected:
        raise ValueError(f"SMPL humanoid body_pose must have shape [..., {expected}], got {tuple(body_pose.shape)}")
    axis_angle = body_pose.reshape(*body_pose.shape[:-1], num_actuated_joints, 3)
    return SO3.conversions.from_axis_angle_to_rotmat(axis_angle, xp=xp)


def forward_skeleton(
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    actuated_joint_indices: list[int],
    parents: list[int],
    body_pose: Float[Array, "B Q"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
    """Compute world-space SMPL humanoid joint transforms."""
    if xp is None:
        xp = get_namespace(body_pose)
    body_rot = _body_rotations(body_pose, len(actuated_joint_indices), xp=xp)
    return rigid.forward_skeleton_from_local_rotations(
        body_rot,
        local_offsets=local_offsets,
        rest_local_rotations=rest_local_rotations,
        actuated_joint_indices=actuated_joint_indices,
        parents=parents,
        global_translation=global_translation,
        global_rotation=global_rotation,
        global_rotation_type=rotation_type,
        joint_indices=joint_indices,
        xp=xp,
    )


def forward_links(
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    actuated_joint_indices: list[int],
    parents: list[int],
    link_joint_indices: list[int],
    link_geom_positions: Float[Array, "L 3"],
    link_geom_rotations: Float[Array, "L 3 3"],
    body_pose: Float[Array, "B Q"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    rotation_type: RotationType = "axis_angle",
    xp: Any = None,
) -> Float[Array, "B L 4 4"]:
    """Compute world-space transforms for each SMPL humanoid link mesh."""
    if xp is None:
        xp = get_namespace(body_pose)
    skeleton = forward_skeleton(
        local_offsets=local_offsets,
        rest_local_rotations=rest_local_rotations,
        actuated_joint_indices=actuated_joint_indices,
        parents=parents,
        body_pose=body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        rotation_type=rotation_type,
        xp=xp,
    )
    return rigid.forward_link_transforms(
        skeleton=skeleton,
        link_joint_indices=link_joint_indices,
        link_geom_positions=link_geom_positions,
        link_geom_rotations=link_geom_rotations,
        xp=xp,
    )


def forward_meshes(
    vertices: Float[Array, "V 3"],
    faces: Int[Array, "F 3"],
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    actuated_joint_indices: list[int],
    parents: list[int],
    link_joint_indices: list[int],
    link_vertex_starts: list[int],
    link_vertex_counts: list[int],
    link_face_starts: list[int],
    link_face_counts: list[int],
    link_geom_positions: Float[Array, "L 3"],
    link_geom_rotations: Float[Array, "L 3 3"],
    body_pose: Float[Array, "B Q"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    rotation_type: RotationType = "axis_angle",
    xp: Any = None,
) -> list[Trimesh]:
    """Rigidly transform and concatenate all SMPL humanoid link meshes."""
    if xp is None:
        xp = get_namespace(body_pose)
    links = forward_links(
        local_offsets=local_offsets,
        rest_local_rotations=rest_local_rotations,
        actuated_joint_indices=actuated_joint_indices,
        parents=parents,
        link_joint_indices=link_joint_indices,
        link_geom_positions=link_geom_positions,
        link_geom_rotations=link_geom_rotations,
        body_pose=body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        rotation_type=rotation_type,
        xp=xp,
    )
    return rigid.forward_meshes_from_links(
        links=links,
        vertices=vertices,
        faces=faces,
        link_vertex_starts=link_vertex_starts,
        link_vertex_counts=link_vertex_counts,
        link_face_starts=link_face_starts,
        link_face_counts=link_face_counts,
        xp=xp,
    )


__all__ = [
    "VALID_ROTATION_TYPES",
    "RotationType",
    "forward_links",
    "forward_meshes",
    "forward_skeleton",
]
