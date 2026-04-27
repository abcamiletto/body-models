"""Backend-agnostic MyoFullBody musculoskeletal kinematics.

The model is parsed from MuJoCo MJCF (:mod:`body_models.myofullbody.io`) into a
body tree where each ``<body>`` becomes a "joint" frame in the body-models API.
Each body owns zero or more 1-DoF MuJoCo joints (``hinge`` or ``slide``) which
are composed in XML order to form the body's local transform.
"""

from __future__ import annotations

from typing import Any

from array_api_compat import get_namespace
from jaxtyping import Float, Int
from nanomanifold import SO3

from .. import common

Array = Any

SKIN_WEIGHTS_ERROR = "MyoFullBody is a rigid articulated model and does not define skin_weights."
# Matches body_models.myofullbody.io.MUJOCO_TO_KIMODO (Ry(+90°) @ kimodo swap)
# so the qpos round-trip recovers the same world frame the loader produces.
MUJOCO_TO_KIMODO = (
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
)


# ----------------------------------------------------------------------------
# Per-joint local transforms
# ----------------------------------------------------------------------------


def _qpos_local_transforms(
    body_pose: Float[Array, "B Q"],
    qpos_axes: Float[Array, "Q 3"],
    qpos_anchors: Float[Array, "Q 3"],
    hinge_mask: Float[Array, "Q"],
    slide_mask: Float[Array, "Q"],
    *,
    xp: Any,
) -> tuple[Float[Array, "B Q 3 3"], Float[Array, "B Q 3"]]:
    """Compute per-qpos local rotation+translation contributed by each joint.

    Returns ``(R, t)`` where the joint transforms the body frame as
    ``T_joint = T(t) @ R``. For hinge joints with anchor ``p`` and angle ``q``:
    ``R = R(axis, q)`` and ``t = p - R @ p``. For slide joints: ``R = I`` and
    ``t = q * axis``.
    """
    angles = body_pose * hinge_mask
    aa = angles[..., None] * qpos_axes[None]
    R = SO3.conversions.from_axis_angle_to_rotmat(aa, xp=xp)

    # For a hinge with anchor p, T_joint = T(p) @ R @ T(-p) collapses to (R, p - R@p).
    p = qpos_anchors[None]
    Rp = xp.squeeze(R @ p[..., None], axis=-1)
    hinge_t = (p - Rp) * hinge_mask[None, :, None]

    slide_t = (body_pose * slide_mask)[..., None] * qpos_axes[None]
    return R, hinge_t + slide_t


# ----------------------------------------------------------------------------
# Forward kinematics
# ----------------------------------------------------------------------------


def forward_skeleton(
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    parents: list[int],
    body_qpos_starts: list[int],
    body_qpos_counts: list[int],
    qpos_axes: Float[Array, "Q 3"],
    qpos_anchors: Float[Array, "Q 3"],
    hinge_mask: Float[Array, "Q"],
    slide_mask: Float[Array, "Q"],
    body_pose: Float[Array, "B Q"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
    """Compute world-space body transforms ``[B, J, 4, 4]`` in meters."""
    if xp is None:
        xp = get_namespace(body_pose)

    num_joints = len(parents)
    if body_pose.ndim != 2 or body_pose.shape[1] != qpos_axes.shape[0]:
        raise ValueError(
            f"body_pose must have shape [B, {qpos_axes.shape[0]}], got {tuple(body_pose.shape)}"
        )

    B = body_pose.shape[0]
    dtype = body_pose.dtype
    if global_translation is None:
        global_translation = common.zeros_as(body_pose, shape=(B, 3), xp=xp)

    qpos_axes = xp.asarray(qpos_axes, dtype=dtype)
    qpos_anchors = xp.asarray(qpos_anchors, dtype=dtype)
    hinge_mask = xp.asarray(hinge_mask, dtype=dtype)
    slide_mask = xp.asarray(slide_mask, dtype=dtype)
    rest_rot = xp.asarray(rest_local_rotations, dtype=dtype)
    rest_t = xp.asarray(local_offsets, dtype=dtype)

    R_q, t_q = _qpos_local_transforms(
        body_pose, qpos_axes, qpos_anchors, hinge_mask, slide_mask, xp=xp
    )

    rot_world: list[Array | None] = [None] * num_joints
    pos_world: list[Array | None] = [None] * num_joints

    for j in range(num_joints):
        local_rot = xp.broadcast_to(rest_rot[j][None], (B, 3, 3))
        local_t = xp.broadcast_to(rest_t[j][None], (B, 3))
        start = body_qpos_starts[j]
        for k in range(start, start + body_qpos_counts[j]):
            local_t = local_t + xp.squeeze(local_rot @ t_q[:, k, :, None], axis=-1)
            local_rot = local_rot @ R_q[:, k]

        if parents[j] < 0:
            rot_world[j] = local_rot
            pos_world[j] = local_t
        else:
            p_rot = rot_world[parents[j]]
            p_pos = pos_world[parents[j]]
            rot_world[j] = p_rot @ local_rot
            pos_world[j] = p_pos + xp.squeeze(p_rot @ local_t[..., None], axis=-1)

    rot = xp.stack(rot_world, axis=1)
    trans = xp.stack(pos_world, axis=1)

    if global_rotation is not None:
        global_rot = SO3.conversions.from_axis_angle_to_rotmat(global_rotation, xp=xp)
        rot = global_rot[:, None] @ rot
        trans = xp.squeeze(global_rot[:, None] @ trans[..., None], axis=-1)
    trans = trans + global_translation[:, None]

    if joint_indices is not None:
        if any(j < 0 or j >= num_joints for j in joint_indices):
            raise IndexError(f"joint_indices must be in [0, {num_joints})")
        rot = rot[:, joint_indices]
        trans = trans[:, joint_indices]

    last_row = common.zeros_as(rot, shape=(B, rot.shape[1], 1, 4), xp=xp)
    last_row = common.set(last_row, (..., 0, 3), xp.asarray(1.0, dtype=dtype), xp=xp)
    return xp.concat([xp.concat([rot, trans[..., None]], axis=-1), last_row], axis=-2)


def forward_links(
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    parents: list[int],
    body_qpos_starts: list[int],
    body_qpos_counts: list[int],
    qpos_axes: Float[Array, "Q 3"],
    qpos_anchors: Float[Array, "Q 3"],
    hinge_mask: Float[Array, "Q"],
    slide_mask: Float[Array, "Q"],
    link_joint_indices: list[int],
    link_geom_positions: Float[Array, "L 3"],
    link_geom_rotations: Float[Array, "L 3 3"],
    body_pose: Float[Array, "B Q"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B 3"] | None = None,
    xp: Any = None,
) -> Float[Array, "B L 4 4"]:
    """Compute world-space transforms for each STL link mesh."""
    if xp is None:
        xp = get_namespace(body_pose)
    skeleton = forward_skeleton(
        local_offsets=local_offsets,
        rest_local_rotations=rest_local_rotations,
        parents=parents,
        body_qpos_starts=body_qpos_starts,
        body_qpos_counts=body_qpos_counts,
        qpos_axes=qpos_axes,
        qpos_anchors=qpos_anchors,
        hinge_mask=hinge_mask,
        slide_mask=slide_mask,
        body_pose=body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        xp=xp,
    )
    joint_rot = skeleton[..., :3, :3]
    joint_pos = skeleton[..., :3, 3]
    geom_pos = xp.asarray(link_geom_positions, dtype=body_pose.dtype)
    geom_rot = xp.asarray(link_geom_rotations, dtype=body_pose.dtype)

    rotations = []
    translations = []
    for link_idx, joint_idx in enumerate(link_joint_indices):
        rotations.append(joint_rot[:, joint_idx] @ geom_rot[link_idx])
        translations.append(
            joint_pos[:, joint_idx] + xp.squeeze(joint_rot[:, joint_idx] @ geom_pos[link_idx][None, :, None], axis=-1)
        )

    rot = xp.stack(rotations, axis=1)
    trans = xp.stack(translations, axis=1)
    last_row = common.zeros_as(rot, shape=(rot.shape[0], rot.shape[1], 1, 4), xp=xp)
    last_row = common.set(last_row, (..., 0, 3), xp.asarray(1.0, dtype=rot.dtype), xp=xp)
    return xp.concat([xp.concat([rot, trans[..., None]], axis=-1), last_row], axis=-2)


def forward_vertices(
    vertices: Float[Array, "V 3"],
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    parents: list[int],
    body_qpos_starts: list[int],
    body_qpos_counts: list[int],
    qpos_axes: Float[Array, "Q 3"],
    qpos_anchors: Float[Array, "Q 3"],
    hinge_mask: Float[Array, "Q"],
    slide_mask: Float[Array, "Q"],
    link_joint_indices: list[int],
    link_vertex_starts: list[int],
    link_vertex_counts: list[int],
    link_geom_positions: Float[Array, "L 3"],
    link_geom_rotations: Float[Array, "L 3 3"],
    body_pose: Float[Array, "B Q"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Rigidly transform per-link STL meshes into the world frame."""
    if xp is None:
        xp = get_namespace(body_pose)
    links = forward_links(
        local_offsets=local_offsets,
        rest_local_rotations=rest_local_rotations,
        parents=parents,
        body_qpos_starts=body_qpos_starts,
        body_qpos_counts=body_qpos_counts,
        qpos_axes=qpos_axes,
        qpos_anchors=qpos_anchors,
        hinge_mask=hinge_mask,
        slide_mask=slide_mask,
        link_joint_indices=link_joint_indices,
        link_geom_positions=link_geom_positions,
        link_geom_rotations=link_geom_rotations,
        body_pose=body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        xp=xp,
    )
    link_rot = links[..., :3, :3]
    link_pos = links[..., :3, 3]
    source_vertices = xp.asarray(vertices, dtype=body_pose.dtype)

    chunks = []
    for link_idx in range(len(link_joint_indices)):
        start = link_vertex_starts[link_idx]
        count = link_vertex_counts[link_idx]
        local_vertices = source_vertices[start : start + count]
        transformed = xp.squeeze(link_rot[:, link_idx, None] @ local_vertices[None, :, :, None], axis=-1)
        transformed = transformed + link_pos[:, link_idx, None]
        chunks.append(transformed)

    out = xp.concat(chunks, axis=1)
    if vertex_indices is not None:
        out = out[:, xp.asarray(vertex_indices)]
    return out


# ----------------------------------------------------------------------------
# Site / tendon helpers (no FK rerun — accept a forward_skeleton output)
# ----------------------------------------------------------------------------


def world_sites(
    skeleton: Float[Array, "B J 4 4"],
    site_positions: Float[Array, "S 3"],
    site_body_indices: list[int],
    *,
    xp: Any = None,
) -> Float[Array, "B S 3"]:
    """Apply each site's parent-body world transform to its body-local position.

    ``skeleton`` is the same ``[B, J, 4, 4]`` array produced by
    :func:`forward_skeleton`. This is just a gather + affine transform — no FK
    is recomputed — so muscle visualisation reuses the existing forward pass.
    """
    if xp is None:
        xp = get_namespace(skeleton)
    body_T = skeleton[:, xp.asarray(site_body_indices)]  # [B, S, 4, 4]
    local = xp.asarray(site_positions, dtype=skeleton.dtype)
    rotated = xp.squeeze(body_T[..., :3, :3] @ local[None, :, :, None], axis=-1)
    return rotated + body_T[..., :3, 3]


# ----------------------------------------------------------------------------
# Mesh helpers
# ----------------------------------------------------------------------------


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
) -> dict[str, Array | str | int]:
    """Return the static STL chunk for one link mesh."""
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
) -> list[dict[str, Array | str | int]]:
    """Return all static STL chunks attached to one body frame."""
    joint_idx = joint_names.index(joint_name)
    meshes = []
    for link_idx, link_name in enumerate(link_names):
        if link_joint_indices[link_idx] != joint_idx:
            continue
        vertex_start = link_vertex_starts[link_idx]
        vertex_count = link_vertex_counts[link_idx]
        face_start = link_face_starts[link_idx]
        face_count = link_face_counts[link_idx]
        meshes.append(
            {
                "name": link_name,
                "vertices": vertices[vertex_start : vertex_start + vertex_count],
                "faces": faces[face_start : face_start + face_count] - vertex_start,
                "joint_index": joint_idx,
                "joint_name": joint_name,
            }
        )
    return meshes


# ----------------------------------------------------------------------------
# MuJoCo qpos round-trip
# ----------------------------------------------------------------------------


def to_mujoco_qpos(
    body_pose: Float[Array, "B Q"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B 3"] | None = None,
    xp: Any = None,
) -> Float[Array, "B 7+Q"]:
    """Build MuJoCo ``qpos`` (free root + joint scalars) from body-models inputs.

    The first 7 entries are the freejoint values (``xyz`` + ``wxyz``); the rest
    are scalar joint coordinates in MJCF order.
    """
    if xp is None:
        xp = get_namespace(body_pose)
    B = body_pose.shape[0]
    dtype = body_pose.dtype
    if global_translation is None:
        global_translation = common.zeros_as(body_pose, shape=(B, 3), xp=xp)
    if global_rotation is None:
        rot_mat = common.eye_as(body_pose, batch_dims=(B,), xp=xp)
    else:
        rot_mat = SO3.conversions.from_axis_angle_to_rotmat(global_rotation, xp=xp)

    coord = xp.asarray(MUJOCO_TO_KIMODO, dtype=dtype)
    kimodo_to_mujoco = coord.mT if hasattr(coord, "mT") else xp.swapaxes(coord, -1, -2)
    root_t = xp.squeeze(kimodo_to_mujoco @ global_translation[..., None], axis=-1)
    root_rot_mujoco = kimodo_to_mujoco[None] @ rot_mat @ coord[None]
    root_quat = SO3.conversions.from_rotmat_to_quat(root_rot_mujoco, convention="wxyz", xp=xp)
    return xp.concat([root_t, root_quat, body_pose], axis=1)


def from_mujoco_qpos(
    qpos: Float[Array, "B 7+Q"],
    *,
    xp: Any = None,
) -> dict[str, Array]:
    """Split a MuJoCo ``qpos`` into ``body_pose`` + global root parameters."""
    if xp is None:
        xp = get_namespace(qpos)
    root_t_mj = qpos[:, :3]
    root_q_mj = qpos[:, 3:7]
    body_pose = qpos[:, 7:]

    coord = xp.asarray(MUJOCO_TO_KIMODO, dtype=qpos.dtype)
    kimodo_to_mujoco = coord.mT if hasattr(coord, "mT") else xp.swapaxes(coord, -1, -2)

    global_translation = xp.squeeze(coord @ root_t_mj[..., None], axis=-1)
    R_mj = SO3.conversions.from_quat_to_rotmat(root_q_mj, convention="wxyz", xp=xp)
    R_k = coord[None] @ R_mj @ kimodo_to_mujoco[None]
    global_rotation = SO3.conversions.from_rotmat_to_axis_angle(R_k, xp=xp)
    return {
        "body_pose": body_pose,
        "global_rotation": global_rotation,
        "global_translation": global_translation,
    }
