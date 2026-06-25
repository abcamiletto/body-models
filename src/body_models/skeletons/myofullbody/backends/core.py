"""Backend-agnostic MyoFullBody musculoskeletal kinematics.

The model is parsed from MuJoCo MJCF (:mod:`body_models.skeletons.myofullbody.io`) into a
body tree where each ``<body>`` becomes a "joint" frame in the body-models API.
Each body owns zero or more 1-DoF MuJoCo joints (``hinge`` or ``slide``) which
are composed in XML order to form the body's local transform.
"""

from __future__ import annotations

from typing import Any

from array_api_compat import get_namespace
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import common
from trimesh import Trimesh
from body_models.common import rigid

Array = Any

# Matches body_models.skeletons.myofullbody.io.MUJOCO_TO_KIMODO (Ry(+90°) @ kimodo swap)
# so the qpos round-trip recovers the same world frame the loader produces.
MUJOCO_TO_KIMODO = (
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
)


# ----------------------------------------------------------------------------
# Per-joint local transforms
# ----------------------------------------------------------------------------


def _actuated_local_transforms(
    body_pose: Float[Array, "B Q"],
    actuated_joint_axes: Float[Array, "Q 3"],
    actuated_joint_anchors: Float[Array, "Q 3"],
    hinge_mask: Float[Array, "Q"],
    slide_mask: Float[Array, "Q"],
    *,
    xp: Any,
) -> tuple[Float[Array, "B Q 3 3"], Float[Array, "B Q 3"]]:
    """Compute per-coordinate local rotation+translation contributed by each joint.

    Returns ``(R, t)`` where the joint transforms the body frame as
    ``T_joint = T(t) @ R``. For hinge joints with anchor ``p`` and angle ``q``:
    ``R = R(axis, q)`` and ``t = p - R @ p``. For slide joints: ``R = I`` and
    ``t = q * axis``.
    """
    angles = body_pose * hinge_mask
    aa = angles[..., None] * actuated_joint_axes
    R = SO3.conversions.from_axis_angle_to_rotmat(aa, xp=xp)

    # For a hinge with anchor p, T_joint = T(p) @ R @ T(-p) collapses to (R, p - R@p).
    p = actuated_joint_anchors
    Rp = xp.squeeze(R @ p[..., None], axis=-1)
    hinge_t = (p - Rp) * hinge_mask[..., None]

    slide_t = (body_pose * slide_mask)[..., None] * actuated_joint_axes
    return R, hinge_t + slide_t


# ----------------------------------------------------------------------------
# Forward kinematics
# ----------------------------------------------------------------------------


def forward_skeleton(
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    parents: list[int],
    body_actuated_starts: list[int],
    body_actuated_counts: list[int],
    actuated_joint_axes: Float[Array, "Q 3"],
    actuated_joint_anchors: Float[Array, "Q 3"],
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
    if body_pose.ndim < 1 or body_pose.shape[-1] != actuated_joint_axes.shape[0]:
        raise ValueError(
            f"body_pose must have shape [..., {actuated_joint_axes.shape[0]}], got {tuple(body_pose.shape)}"
        )

    batch_shape = tuple(body_pose.shape[:-1])
    dtype = body_pose.dtype
    if global_translation is None:
        global_translation = common.zeros_as(body_pose, shape=(*batch_shape, 3), xp=xp)

    actuated_joint_axes = xp.asarray(actuated_joint_axes, dtype=dtype)
    actuated_joint_anchors = xp.asarray(actuated_joint_anchors, dtype=dtype)
    hinge_mask = xp.asarray(hinge_mask, dtype=dtype)
    slide_mask = xp.asarray(slide_mask, dtype=dtype)
    rest_rot = xp.asarray(rest_local_rotations, dtype=dtype)
    rest_t = xp.asarray(local_offsets, dtype=dtype)

    actuated_rot, actuated_trans = _actuated_local_transforms(
        body_pose,
        actuated_joint_axes,
        actuated_joint_anchors,
        hinge_mask,
        slide_mask,
        xp=xp,
    )

    rot_world: list[Array | None] = [None] * num_joints
    pos_world: list[Array | None] = [None] * num_joints

    for j in range(num_joints):
        local_rot = xp.broadcast_to(rest_rot[j], (*batch_shape, 3, 3))
        local_t = xp.broadcast_to(rest_t[j], (*batch_shape, 3))
        start = body_actuated_starts[j]
        for k in range(start, start + body_actuated_counts[j]):
            actuated_t = xp.squeeze(local_rot @ actuated_trans[..., k, :, None], axis=-1)
            local_t = local_t + actuated_t
            local_rot = local_rot @ actuated_rot[..., k, :, :]

        if parents[j] < 0:
            rot_world[j] = local_rot
            pos_world[j] = local_t
        else:
            p_rot = rot_world[parents[j]]
            p_pos = pos_world[parents[j]]
            rot_world[j] = p_rot @ local_rot
            local_pos = xp.squeeze(p_rot @ local_t[..., None], axis=-1)
            pos_world[j] = p_pos + local_pos

    rot = xp.stack(rot_world, axis=-3)
    trans = xp.stack(pos_world, axis=-2)

    if global_rotation is not None:
        global_rot = SO3.conversions.from_axis_angle_to_rotmat(global_rotation, xp=xp)
        rot = global_rot[..., None, :, :] @ rot
        trans = xp.squeeze(global_rot[..., None, :, :] @ trans[..., None], axis=-1)
    trans = trans + global_translation[..., None, :]

    if joint_indices is not None:
        if any(j < 0 or j >= num_joints for j in joint_indices):
            raise IndexError(f"joint_indices must be in [0, {num_joints})")
        rot = rot[..., joint_indices, :, :]
        trans = trans[..., joint_indices, :]

    last_row = common.zeros_as(rot, shape=(*rot.shape[:-2], 1, 4), xp=xp)
    last_row = common.set(last_row, (..., 0, 3), xp.asarray(1.0, dtype=dtype), xp=xp)
    return xp.concat([xp.concat([rot, trans[..., None]], axis=-1), last_row], axis=-2)


def forward_links(
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    parents: list[int],
    body_actuated_starts: list[int],
    body_actuated_counts: list[int],
    actuated_joint_axes: Float[Array, "Q 3"],
    actuated_joint_anchors: Float[Array, "Q 3"],
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
        body_actuated_starts=body_actuated_starts,
        body_actuated_counts=body_actuated_counts,
        actuated_joint_axes=actuated_joint_axes,
        actuated_joint_anchors=actuated_joint_anchors,
        hinge_mask=hinge_mask,
        slide_mask=slide_mask,
        body_pose=body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
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
    parents: list[int],
    body_actuated_starts: list[int],
    body_actuated_counts: list[int],
    actuated_joint_axes: Float[Array, "Q 3"],
    actuated_joint_anchors: Float[Array, "Q 3"],
    hinge_mask: Float[Array, "Q"],
    slide_mask: Float[Array, "Q"],
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
    global_rotation: Float[Array, "B 3"] | None = None,
    xp: Any = None,
) -> list[Trimesh]:
    """Rigidly transform and concatenate all MyoFullBody STL link meshes."""
    if xp is None:
        xp = get_namespace(body_pose)
    links = forward_links(
        local_offsets=local_offsets,
        rest_local_rotations=rest_local_rotations,
        parents=parents,
        body_actuated_starts=body_actuated_starts,
        body_actuated_counts=body_actuated_counts,
        actuated_joint_axes=actuated_joint_axes,
        actuated_joint_anchors=actuated_joint_anchors,
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

    ``skeleton`` is the same ``[..., J, 4, 4]`` array produced by
    :func:`forward_skeleton`. This is just a gather + affine transform — no FK
    is recomputed — so muscle visualisation reuses the existing forward pass.
    """
    if xp is None:
        xp = get_namespace(skeleton)
    body_T = skeleton[..., xp.asarray(site_body_indices), :, :]
    local = xp.asarray(site_positions, dtype=skeleton.dtype)
    rotated = xp.squeeze(body_T[..., :3, :3] @ local[..., None], axis=-1)
    return rotated + body_T[..., :3, 3]


# ----------------------------------------------------------------------------
# MuJoCo qpos round-trip
# ----------------------------------------------------------------------------


def from_mujoco_qpos(
    qpos: Float[Array, "B 7+Q"],
    *,
    xp: Any = None,
) -> dict[str, Array]:
    """Split a MuJoCo ``qpos`` into ``body_pose`` + global root parameters."""
    if xp is None:
        xp = get_namespace(qpos)
    root_t_mj = qpos[..., :3]
    root_q_mj = qpos[..., 3:7]
    body_pose = qpos[..., 7:]

    coord = xp.asarray(MUJOCO_TO_KIMODO, dtype=qpos.dtype)
    kimodo_to_mujoco = coord.mT if hasattr(coord, "mT") else xp.swapaxes(coord, -1, -2)

    global_translation = xp.squeeze(coord @ root_t_mj[..., None], axis=-1)
    R_mj = SO3.conversions.from_quat_to_rotmat(root_q_mj, convention="wxyz", xp=xp)
    R_k = coord @ R_mj @ kimodo_to_mujoco
    global_rotation = SO3.conversions.from_rotmat_to_axis_angle(R_k, xp=xp)
    return {
        "body_pose": body_pose,
        "global_rotation": global_rotation,
        "global_translation": global_translation,
    }
