"""Backend-agnostic Unitree G1 rigid articulated model computation."""

from __future__ import annotations

from typing import Any, Literal

from array_api_compat import get_namespace
from jaxtyping import Float, Int
from nanomanifold import SO3

from trimesh import Trimesh
from body_models.common import rigid
from body_models.rotations import RotationType as SO3RotationType

Array = Any
RotationType = SO3RotationType | Literal["hinge"]
Convention = Literal["soma", "mujoco"]

MUJOCO_TO_KIMODO = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
VALID_ROTATION_TYPES = ("axis_angle", "quat", "sixd", "matrix", "rotmat", "hinge")
GLOBAL_ROTATION_TYPES: dict[RotationType, SO3RotationType] = {
    "axis_angle": "axis_angle",
    "quat": "quat",
    "sixd": "sixd",
    "matrix": "matrix",
    "rotmat": "rotmat",
    "hinge": "rotmat",
}


def _hinge_rotations(
    body_pose: Float[Array, "B Q"],
    actuated_joint_axes: Float[Array, "Q 3"],
    *,
    xp: Any,
) -> Float[Array, "B Q 3 3"]:
    if body_pose.ndim < 1 or body_pose.shape[-1] != actuated_joint_axes.shape[0]:
        raise ValueError(
            f"G1 body_pose must have shape [..., {actuated_joint_axes.shape[0]}], got {tuple(body_pose.shape)}"
        )
    axes = xp.asarray(actuated_joint_axes, dtype=body_pose.dtype)
    return SO3.convert(body_pose[..., None], src="hinge", dst="rotmat", src_kwargs={"axes": axes}, xp=xp)


def forward_skeleton(
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    actuated_joint_indices: list[int],
    actuated_joint_axes: Float[Array, "Q 3"],
    parents: list[int],
    body_pose: Float[Array, "B Q"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "rotmat",
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
    """Compute world-space G1 joint transforms from local rotations."""
    if xp is None:
        xp = get_namespace(body_pose)
    body_rot = _hinge_rotations(body_pose, actuated_joint_axes, xp=xp)
    return rigid.forward_skeleton_from_local_rotations(
        body_rot,
        local_offsets=local_offsets,
        rest_local_rotations=rest_local_rotations,
        actuated_joint_indices=actuated_joint_indices,
        parents=parents,
        global_translation=global_translation,
        global_rotation=global_rotation,
        global_rotation_type=GLOBAL_ROTATION_TYPES[rotation_type],
        joint_indices=joint_indices,
        xp=xp,
    )


def forward_meshes(
    vertices: Float[Array, "V 3"],
    faces: Int[Array, "F 3"],
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    actuated_joint_indices: list[int],
    actuated_joint_axes: Float[Array, "Q 3"],
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
    rotation_type: RotationType = "rotmat",
    xp: Any = None,
) -> list[Trimesh]:
    """Rigidly transform and concatenate all G1 STL link meshes."""
    if xp is None:
        xp = get_namespace(body_pose)
    links = forward_links(
        local_offsets=local_offsets,
        rest_local_rotations=rest_local_rotations,
        actuated_joint_indices=actuated_joint_indices,
        actuated_joint_axes=actuated_joint_axes,
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


def forward_links(
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    actuated_joint_indices: list[int],
    actuated_joint_axes: Float[Array, "Q 3"],
    parents: list[int],
    link_joint_indices: list[int],
    link_geom_positions: Float[Array, "L 3"],
    link_geom_rotations: Float[Array, "L 3 3"],
    body_pose: Float[Array, "B Q"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    rotation_type: RotationType = "rotmat",
    xp: Any = None,
) -> Float[Array, "B L 4 4"]:
    """Compute world-space transforms for each STL link mesh."""
    if xp is None:
        xp = get_namespace(body_pose)
    skeleton = forward_skeleton(
        local_offsets=local_offsets,
        rest_local_rotations=rest_local_rotations,
        actuated_joint_indices=actuated_joint_indices,
        actuated_joint_axes=actuated_joint_axes,
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
