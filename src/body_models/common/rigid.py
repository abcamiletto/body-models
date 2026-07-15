"""Shared rigid articulated mesh helpers."""

from typing import Any

from jaxtyping import Float, Int
from nanomanifold import SO3
import numpy as np
from trimesh import Trimesh
from trimesh.util import concatenate

from body_models.common.kinematics import affine_transforms
from body_models.common.ops import eye_as, set, zeros_as
from body_models.rotations import RotationType

Array = Any


def rotate_transforms(
    transforms: Float[Array, "... J 4 4"],
    rotation: Float[Array, "... N"] | Float[Array, "... 3 3"] | None,
    rotation_type: RotationType,
    xp: Any,
) -> Float[Array, "... J 4 4"]:
    """Rotate world-space transforms about the origin."""
    if rotation is None:
        return transforms
    rotation_matrix = SO3.convert(rotation, src=rotation_type, dst="rotmat", xp=xp)
    rotated = rotation_matrix[..., None, :, :] @ transforms[..., :3, :]
    return set(transforms, (..., slice(None, 3), slice(None)), rotated, xp=xp)


def forward_skeleton_from_local_rotations(
    body_rotations: Float[Array, "... Q 3 3"],
    *,
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    actuated_joint_indices: list[int],
    parents: list[int],
    global_translation: Float[Array, "... 3"] | None = None,
    global_rotation: Float[Array, "... 3"] | None = None,
    joint_indices: list[int] | None = None,
    xp: Any,
) -> Float[Array, "... J 4 4"]:
    """Compute rigid hierarchy transforms from local actuated joint rotations."""
    batch_shape = tuple(body_rotations.shape[:-3])
    dtype = body_rotations.dtype
    num_joints = len(parents)

    rest_rot = xp.asarray(rest_local_rotations, dtype=dtype)
    local_rot = eye_as(body_rotations, batch_dims=(*batch_shape, num_joints), xp=xp)
    local_rot = set(local_rot, (..., actuated_joint_indices, slice(None), slice(None)), body_rotations, xp=xp)
    local_rot = xp.broadcast_to(rest_rot, (*batch_shape, num_joints, 3, 3)) @ local_rot
    return forward_skeleton_from_local_transforms(
        local_rot,
        local_offsets=local_offsets,
        parents=parents,
        global_translation=global_translation,
        global_rotation=global_rotation,
        joint_indices=joint_indices,
        xp=xp,
    )


def forward_skeleton_from_local_transforms(
    local_rotations: Float[Array, "... J 3 3"],
    *,
    local_offsets: Float[Array, "J 3"],
    parents: list[int],
    global_translation: Float[Array, "... 3"] | None = None,
    global_rotation: Float[Array, "... 3"] | None = None,
    joint_indices: list[int] | None = None,
    xp: Any,
) -> Float[Array, "... J 4 4"]:
    """Compute rigid hierarchy transforms from local joint transforms."""
    batch_shape = tuple(local_rotations.shape[:-3])
    dtype = local_rotations.dtype
    num_joints = len(parents)
    if global_translation is None:
        global_translation = zeros_as(local_rotations, shape=(*batch_shape, 3), xp=xp)

    local_rot = local_rotations
    local_t = xp.asarray(local_offsets, dtype=dtype)

    rot_world: list[Array | None] = [None] * num_joints
    pos_world: list[Array | None] = [None] * num_joints
    rot_world[0] = local_rot[..., 0, :, :]
    pos_world[0] = zeros_as(local_rot, shape=(*batch_shape, 3), xp=xp)
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
        global_rot = SO3.convert(global_rotation, src="axis_angle", dst="rotmat", xp=xp)
        rot = global_rot[..., None, :, :] @ rot
        trans = xp.squeeze(global_rot[..., None, :, :] @ trans[..., None], axis=-1)
    trans = trans + global_translation[..., None, :]

    if joint_indices is not None:
        if any(joint < 0 or joint >= num_joints for joint in joint_indices):
            raise IndexError(f"joint_indices must be in [0, {num_joints})")
        rot = rot[..., joint_indices, :, :]
        trans = trans[..., joint_indices, :]

    return affine_transforms(rot, trans, xp=xp)


def forward_link_transforms(
    skeleton: Float[Array, "... J 4 4"],
    link_joint_indices: list[int],
    link_geom_positions: Float[Array, "L 3"],
    link_geom_rotations: Float[Array, "L 3 3"],
    *,
    xp: Any,
) -> Float[Array, "... L 4 4"]:
    """Apply each link's local geom transform to its parent joint transform."""
    joint_rot = skeleton[..., :3, :3]
    joint_pos = skeleton[..., :3, 3]
    geom_pos = xp.asarray(link_geom_positions, dtype=skeleton.dtype)
    geom_rot = xp.asarray(link_geom_rotations, dtype=skeleton.dtype)

    rotations = []
    translations = []
    for link_idx, joint_idx in enumerate(link_joint_indices):
        link_rot = joint_rot[..., joint_idx, :, :]
        link_pos = xp.squeeze(link_rot @ geom_pos[link_idx][..., None], axis=-1)
        rotations.append(link_rot @ geom_rot[link_idx])
        translations.append(joint_pos[..., joint_idx, :] + link_pos)

    rot = xp.stack(rotations, axis=-3)
    trans = xp.stack(translations, axis=-2)
    return affine_transforms(rot, trans, xp=xp)


def forward_meshes_from_links(
    links: Float[Array, "... L 4 4"],
    vertices: Float[Array, "V 3"],
    faces: Int[Array, "F 3"],
    link_vertex_starts: list[int],
    link_vertex_counts: list[int],
    link_face_starts: list[int],
    link_face_counts: list[int],
    *,
    xp: Any,
) -> list[Trimesh]:
    """Build one concatenated ``Trimesh`` per batch element."""
    link_rot = links[..., :3, :3]
    link_pos = links[..., :3, 3]
    source_vertices = xp.asarray(vertices, dtype=links.dtype)
    source_faces = xp.asarray(faces)
    batch_size = _batch_size(links)
    meshes_by_batch: list[list[Trimesh]] = [[] for _ in range(batch_size)]

    for link_idx in range(len(link_vertex_starts)):
        vertex_start = link_vertex_starts[link_idx]
        vertex_count = link_vertex_counts[link_idx]
        face_start = link_face_starts[link_idx]
        face_count = link_face_counts[link_idx]
        local_vertices = source_vertices[vertex_start : vertex_start + vertex_count]
        transformed = xp.squeeze(link_rot[..., link_idx, None, :, :] @ local_vertices[..., None], axis=-1)
        mesh_vertices = transformed + link_pos[..., link_idx, None, :]
        mesh_faces = source_faces[face_start : face_start + face_count] - vertex_start
        batched_vertices = _as_batched_vertices(mesh_vertices, batch_size=batch_size)
        faces_np = _as_numpy(mesh_faces)
        for batch_idx, batch_vertices in enumerate(batched_vertices):
            meshes_by_batch[batch_idx].append(_make_trimesh(vertices=batch_vertices, faces=faces_np))

    return [_concatenate_meshes(batch_meshes) for batch_meshes in meshes_by_batch]


def _make_trimesh(
    *,
    vertices: Float[Array, "... V 3"],
    faces: Int[Array, "F 3"],
) -> Trimesh:
    return Trimesh(
        vertices=_as_unbatched_vertices(vertices),
        faces=_as_numpy(faces),
        process=False,
    )


def _as_unbatched_vertices(vertices: Float[Array, "... V 3"]) -> Float[np.ndarray, "V 3"]:
    vertices = _as_numpy(vertices)
    if vertices.ndim > 2 and int(np.prod(vertices.shape[:-2])) == 1:
        vertices = vertices.reshape(vertices.shape[-2], vertices.shape[-1])
    if vertices.ndim != 2:
        raise ValueError("Trimesh construction only supports unbatched vertices.")
    return vertices


def _as_batched_vertices(vertices: Float[Array, "... V 3"], *, batch_size: int) -> Float[np.ndarray, "B V 3"]:
    vertices = _as_numpy(vertices)
    if vertices.ndim == 2:
        vertices = vertices[None]
    if vertices.ndim < 3:
        raise ValueError("forward_meshes expects vertices with shape [..., V, 3].")
    return vertices.reshape(batch_size, vertices.shape[-2], vertices.shape[-1])


def _batch_size(links: Float[Array, "... L 4 4"]) -> int:
    batch_shape = links.shape[:-3]
    if not batch_shape:
        return 1
    return int(np.prod(batch_shape))


def _concatenate_meshes(meshes: list[Trimesh]) -> Trimesh:
    if not meshes:
        return Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64), process=False)
    return concatenate(meshes)


def _as_numpy(value: Any) -> Float[np.ndarray, "..."] | Int[np.ndarray, "..."]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)
