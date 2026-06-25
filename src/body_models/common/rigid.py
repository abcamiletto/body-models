"""Shared rigid articulated mesh helpers."""

from typing import Any

from jaxtyping import Float, Int

from body_models.base import MeshPayload
from body_models.common.ops import set, zeros_as

Array = Any


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
    last_row = zeros_as(rot, shape=(*rot.shape[:-2], 1, 4), xp=xp)
    last_row = set(last_row, (..., 0, 3), xp.asarray(1.0, dtype=rot.dtype), xp=xp)
    return xp.concat([xp.concat([rot, trans[..., None]], axis=-1), last_row], axis=-2)


def forward_meshes_from_links(
    links: Float[Array, "... L 4 4"],
    vertices: Float[Array, "V 3"],
    faces: Int[Array, "F 3"],
    link_joint_indices: list[int],
    link_vertex_starts: list[int],
    link_vertex_counts: list[int],
    link_face_starts: list[int],
    link_face_counts: list[int],
    link_names: list[str],
    joint_names: list[str],
    *,
    link_indices: list[int] | None = None,
    xp: Any,
) -> list[MeshPayload]:
    """Build one transformed mesh payload per requested rigid link."""
    link_rot = links[..., :3, :3]
    link_pos = links[..., :3, 3]
    source_vertices = xp.asarray(vertices, dtype=links.dtype)
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
    """Return the static mesh chunk for one rigid link."""
    link_idx = link_names.index(link_name)
    vertex_start = link_vertex_starts[link_idx]
    vertex_count = link_vertex_counts[link_idx]
    face_start = link_face_starts[link_idx]
    face_count = link_face_counts[link_idx]
    joint_idx = link_joint_indices[link_idx]
    return {
        "name": link_name,
        "vertices": vertices[vertex_start : vertex_start + vertex_count],
        "faces": faces[face_start : face_start + face_count] - vertex_start,
        "joint_index": joint_idx,
        "joint_name": joint_names[joint_idx],
    }


def joint_meshes(
    vertices: Float[Array, "V 3"],
    faces: Int[Array, "F 3"],
    link_joint_indices: list[int],
    link_vertex_starts: list[int],
    link_vertex_counts: list[int],
    link_face_starts: list[int],
    link_face_counts: list[int],
    joint_names: list[str],
    link_names: list[str],
    joint_name: str,
) -> list[MeshPayload]:
    """Return all static mesh chunks attached to one rigid joint."""
    joint_idx = joint_names.index(joint_name)
    meshes = []
    for link_idx, link_name in enumerate(link_names):
        if link_joint_indices[link_idx] != joint_idx:
            continue
        meshes.append(
            link_mesh(
                vertices=vertices,
                faces=faces,
                link_joint_indices=link_joint_indices,
                link_vertex_starts=link_vertex_starts,
                link_vertex_counts=link_vertex_counts,
                link_face_starts=link_face_starts,
                link_face_counts=link_face_counts,
                joint_names=joint_names,
                link_names=link_names,
                link_name=link_name,
            )
        )
    return meshes
