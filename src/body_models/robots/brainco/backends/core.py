"""Backend-agnostic BrainCo Revo 2 rigid hand computation."""

from __future__ import annotations

from typing import Any, Literal

from array_api_compat import get_namespace
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import common
from body_models.base import MeshPayload
from body_models.rotations import RotationType as SO3RotationType

Array = Any
RotationType = SO3RotationType | Literal["hinge"]
VALID_ROTATION_TYPES = ("axis_angle", "quat", "sixd", "matrix", "rotmat", "hinge")
GLOBAL_ROTATION_TYPES: dict[RotationType, SO3RotationType] = {
    "axis_angle": "axis_angle",
    "quat": "quat",
    "sixd": "sixd",
    "matrix": "matrix",
    "rotmat": "rotmat",
    "hinge": "rotmat",
}


def forward_skeleton(
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    joint_axes: Float[Array, "Q 3"],
    joint_indices: list[int],
    coupled_joint_axes: Float[Array, "C 3"],
    coupled_joint_indices: list[int],
    coupled_driver_indices: list[int],
    coupled_polycoef: Float[Array, "C 4"],
    parents: list[int],
    pose: Float[Array, "B Q N"] | Float[Array, "B Q 3 3"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    skeleton_indices: list[int] | None = None,
    rotation_type: RotationType = "rotmat",
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
    """Compute world-space BrainCo hand joint transforms."""
    if xp is None:
        xp = get_namespace(pose)
    axes = xp.asarray(joint_axes, dtype=pose.dtype)
    src_kwargs = {"axes": axes} if rotation_type == "hinge" else {}
    local_joint_rot = SO3.convert(pose, src=rotation_type, dst="rotmat", src_kwargs=src_kwargs, xp=xp)
    if local_joint_rot.shape[-2:] != (3, 3):
        raise ValueError("BrainCo pose must convert to shape [..., Q, 3, 3]")

    batch_shape = tuple(local_joint_rot.shape[:-3])
    dtype = local_joint_rot.dtype
    num_joints = len(parents)
    if global_translation is None:
        global_translation = common.zeros_as(local_joint_rot, shape=(*batch_shape, 3), xp=xp)

    rest_rot = xp.asarray(rest_local_rotations, dtype=dtype)
    local_rot = common.eye_as(local_joint_rot, batch_dims=(*batch_shape, num_joints), xp=xp)
    local_rot = common.set(local_rot, (..., joint_indices, slice(None), slice(None)), local_joint_rot, xp=xp)
    if rotation_type == "hinge" and coupled_joint_indices:
        driver_pose = pose[..., coupled_driver_indices, 0]
        coeffs = xp.asarray(coupled_polycoef, dtype=dtype)
        coupled_pose = (
            coeffs[:, 0]
            + coeffs[:, 1] * driver_pose
            + coeffs[:, 2] * driver_pose * driver_pose
            + coeffs[:, 3] * driver_pose * driver_pose * driver_pose
        )
        coupled_rot = SO3.convert(
            coupled_pose[..., None],
            src="hinge",
            dst="rotmat",
            src_kwargs={"axes": xp.asarray(coupled_joint_axes, dtype=dtype)},
            xp=xp,
        )
        local_rot = common.set(local_rot, (..., coupled_joint_indices, slice(None), slice(None)), coupled_rot, xp=xp)
    local_rot = xp.broadcast_to(rest_rot, (*batch_shape, num_joints, 3, 3)) @ local_rot
    local_t = xp.asarray(local_offsets, dtype=dtype)

    rot_world: list[Array | None] = [None] * num_joints
    pos_world: list[Array | None] = [None] * num_joints
    rot_world[0] = local_rot[..., 0, :, :]
    pos_world[0] = common.zeros_as(local_rot, shape=(*batch_shape, 3), xp=xp)
    for joint in range(1, num_joints):
        parent = parents[joint]
        parent_rot = rot_world[parent]
        parent_pos = pos_world[parent]
        rot_world[joint] = parent_rot @ local_rot[..., joint, :, :]
        local_pos = xp.squeeze(parent_rot @ local_t[joint][..., None], axis=-1)
        pos_world[joint] = parent_pos + local_pos

    rot = xp.stack(rot_world, axis=-3)
    trans = xp.stack(pos_world, axis=-2)
    if global_rotation is not None:
        global_rotation_type = GLOBAL_ROTATION_TYPES[rotation_type]
        global_rot = SO3.convert(global_rotation, src=global_rotation_type, dst="rotmat", xp=xp)
        rot = global_rot[..., None, :, :] @ rot
        trans = xp.squeeze(global_rot[..., None, :, :] @ trans[..., None], axis=-1)
    trans = trans + global_translation[..., None, :]

    if skeleton_indices is not None:
        if any(joint < 0 or joint >= num_joints for joint in skeleton_indices):
            raise IndexError(f"skeleton_indices must be in [0, {num_joints})")
        rot = rot[..., skeleton_indices, :, :]
        trans = trans[..., skeleton_indices, :]

    last_row = common.zeros_as(rot, shape=(*rot.shape[:-2], 1, 4), xp=xp)
    last_row = common.set(last_row, (..., 0, 3), xp.asarray(1.0, dtype=dtype), xp=xp)
    return xp.concat([xp.concat([rot, trans[..., None]], axis=-1), last_row], axis=-2)


def forward_links(
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    joint_axes: Float[Array, "Q 3"],
    joint_indices: list[int],
    coupled_joint_axes: Float[Array, "C 3"],
    coupled_joint_indices: list[int],
    coupled_driver_indices: list[int],
    coupled_polycoef: Float[Array, "C 4"],
    parents: list[int],
    link_joint_indices: list[int],
    link_geom_positions: Float[Array, "L 3"],
    link_geom_rotations: Float[Array, "L 3 3"],
    pose: Float[Array, "B Q N"] | Float[Array, "B Q 3 3"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    rotation_type: RotationType = "rotmat",
    xp: Any = None,
) -> Float[Array, "B L 4 4"]:
    """Compute world-space transforms for each BrainCo STL link mesh."""
    if xp is None:
        xp = get_namespace(pose)
    skeleton = forward_skeleton(
        local_offsets=local_offsets,
        rest_local_rotations=rest_local_rotations,
        joint_axes=joint_axes,
        joint_indices=joint_indices,
        coupled_joint_axes=coupled_joint_axes,
        coupled_joint_indices=coupled_joint_indices,
        coupled_driver_indices=coupled_driver_indices,
        coupled_polycoef=coupled_polycoef,
        parents=parents,
        pose=pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        rotation_type=rotation_type,
        xp=xp,
    )
    joint_rot = skeleton[..., :3, :3]
    joint_pos = skeleton[..., :3, 3]
    geom_pos = xp.asarray(link_geom_positions, dtype=pose.dtype)
    geom_rot = xp.asarray(link_geom_rotations, dtype=pose.dtype)

    rotations = []
    translations = []
    for link_idx, joint_idx in enumerate(link_joint_indices):
        link_rot = joint_rot[..., joint_idx, :, :]
        link_pos = xp.squeeze(link_rot @ geom_pos[link_idx][..., None], axis=-1)
        rotations.append(link_rot @ geom_rot[link_idx])
        translations.append(joint_pos[..., joint_idx, :] + link_pos)

    rot = xp.stack(rotations, axis=-3)
    trans = xp.stack(translations, axis=-2)
    last_row = common.zeros_as(rot, shape=(*rot.shape[:-2], 1, 4), xp=xp)
    last_row = common.set(last_row, (..., 0, 3), xp.asarray(1.0, dtype=rot.dtype), xp=xp)
    return xp.concat([xp.concat([rot, trans[..., None]], axis=-1), last_row], axis=-2)


def forward_meshes(
    vertices: Float[Array, "V 3"],
    faces: Int[Array, "F 3"],
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    joint_axes: Float[Array, "Q 3"],
    joint_indices: list[int],
    coupled_joint_axes: Float[Array, "C 3"],
    coupled_joint_indices: list[int],
    coupled_driver_indices: list[int],
    coupled_polycoef: Float[Array, "C 4"],
    parents: list[int],
    link_joint_indices: list[int],
    link_vertex_starts: list[int],
    link_vertex_counts: list[int],
    link_face_starts: list[int],
    link_face_counts: list[int],
    link_geom_positions: Float[Array, "L 3"],
    link_geom_rotations: Float[Array, "L 3 3"],
    link_names: list[str],
    joint_names: list[str],
    pose: Float[Array, "B Q N"] | Float[Array, "B Q 3 3"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    link_indices: list[int] | None = None,
    rotation_type: RotationType = "rotmat",
    xp: Any = None,
) -> list[MeshPayload]:
    """Rigidly transform each BrainCo STL link mesh and keep link boundaries."""
    if xp is None:
        xp = get_namespace(pose)
    links = forward_links(
        local_offsets=local_offsets,
        rest_local_rotations=rest_local_rotations,
        joint_axes=joint_axes,
        joint_indices=joint_indices,
        coupled_joint_axes=coupled_joint_axes,
        coupled_joint_indices=coupled_joint_indices,
        coupled_driver_indices=coupled_driver_indices,
        coupled_polycoef=coupled_polycoef,
        parents=parents,
        link_joint_indices=link_joint_indices,
        link_geom_positions=link_geom_positions,
        link_geom_rotations=link_geom_rotations,
        pose=pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        rotation_type=rotation_type,
        xp=xp,
    )
    link_rot = links[..., :3, :3]
    link_pos = links[..., :3, 3]
    source_vertices = xp.asarray(vertices, dtype=pose.dtype)
    source_faces = xp.asarray(faces)
    indices = range(len(link_names)) if link_indices is None else link_indices

    meshes = []
    for link_idx in indices:
        vertex_start = link_vertex_starts[link_idx]
        vertex_count = link_vertex_counts[link_idx]
        face_start = link_face_starts[link_idx]
        face_count = link_face_counts[link_idx]
        local_vertices = source_vertices[vertex_start : vertex_start + vertex_count]
        transformed = xp.squeeze(link_rot[..., link_idx, None, :, :] @ local_vertices[..., None], axis=-1)
        joint_idx = link_joint_indices[link_idx]
        meshes.append(
            {
                "name": link_names[link_idx],
                "vertices": transformed + link_pos[..., link_idx, None, :],
                "faces": source_faces[face_start : face_start + face_count] - vertex_start,
                "joint_index": joint_idx,
                "joint_name": joint_names[joint_idx],
            }
        )
    return meshes


def link_mesh(
    vertices: Float[Array, "V 3"],
    faces: Int[Array, "F 3"],
    link_joint_indices: list[int],
    link_vertex_starts: list[int],
    link_vertex_counts: list[int],
    link_face_starts: list[int],
    link_face_counts: list[int],
    joint_names: list[str],
    link_names: list[str],
    link_name: str,
) -> MeshPayload:
    if link_name not in link_names:
        raise KeyError(f"Unknown BrainCo link mesh: {link_name}")
    idx = link_names.index(link_name)
    vertex_start = link_vertex_starts[idx]
    vertex_count = link_vertex_counts[idx]
    face_start = link_face_starts[idx]
    face_count = link_face_counts[idx]
    return {
        "name": link_name,
        "vertices": vertices[vertex_start : vertex_start + vertex_count],
        "faces": faces[face_start : face_start + face_count] - vertex_start,
        "joint_index": link_joint_indices[idx],
        "joint_name": joint_names[link_joint_indices[idx]],
    }
