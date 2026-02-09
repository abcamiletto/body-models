"""Backend-agnostic SKEL computation using array_api_compat.

Note: Skeleton mesh computation is NOT included here - it is PyTorch-only.
The torch.py backend adds forward_skeleton_mesh on top of this core computation.
"""

import math
from typing import Any

from array_api_compat import get_namespace
from jaxtyping import Float, Int
from nanomanifold import SO3

from .. import common

Array = Any  # Generic array type (numpy, torch, jax)

# SKEL uses SMPL-compatible pose blend shapes - this maps SKEL joints to SMPL joints
SMPL_JOINT_MAP = [0, 2, 5, 8, 8, 11, 1, 4, 7, 7, 10, 3, 6, 15, 14, 17, 19, 0, 21, 13, 16, 18, 0, 20]

# Constants
NUM_JOINTS = 24
NUM_POSE_PARAMS = 46
NUM_BETAS = 10


def forward_vertices(
    # Model data
    v_template: Float[Array, "V 3"],
    v_template_full: Float[Array, "V_full 3"],
    shapedirs: Float[Array, "V 3 B"],
    shapedirs_full: Float[Array, "V_full 3 B"],
    posedirs: Float[Array, "V*3 P"],
    skin_weights: Float[Array, "V 24"],
    J_regressor: Float[Array, "24 V_full"],
    parents: Int[Array, "23"],
    all_axes: Float[Array, "47 3"],
    rotation_indices: Int[Array, "24 3"],
    apose_R: Float[Array, "24 3 3"],
    apose_t: Float[Array, "24 3"],
    per_joint_rot: Float[Array, "24 3 3"],
    child: Int[Array, "24"],
    fixed_orientation_joints: Int[Array, "6"],
    feet_offset: Float[Array, "3"],
    num_joints_smpl: int,
    scapula_r_axes: Float[Array, "3 3"],
    scapula_l_axes: Float[Array, "3 3"],
    spine_axes: Float[Array, "3 3"],
    # Inputs
    shape: Float[Array, "B 10"],
    pose: Float[Array, "B 46"],
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3]."""
    assert shape.ndim == 2 and shape.shape[1] >= 1
    assert pose.ndim == 2 and pose.shape[1] == NUM_POSE_PARAMS
    assert global_rotation is None or (global_rotation.ndim == 2 and global_rotation.shape[1] == 3)
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(pose)
    B = pose.shape[0]
    dtype = pose.dtype
    Nv = v_template.shape[0]

    if global_translation is None:
        global_translation = xp.zeros((B, 3), dtype=dtype)
    if shape.shape[0] == 1 and B > 1:
        shape = xp.broadcast_to(shape, (B, shape.shape[1]))

    # Joint positions (use full-resolution for accurate skeleton)
    v_shaped_full = v_template_full + xp.einsum("vdi,bi->bvd", shapedirs_full, shape)
    J = xp.einsum("bvd,jv->bjd", v_shaped_full, J_regressor)
    J_rel = _compute_J_rel(xp, J, parents)

    # Forward kinematics
    G_local = _compute_local_transforms(
        xp=xp,
        pose=pose,
        J=J,
        J_rel=J_rel,
        all_axes=all_axes,
        rotation_indices=rotation_indices,
        apose_R=apose_R,
        apose_t=apose_t,
        per_joint_rot=per_joint_rot,
        child=child,
        fixed_orientation_joints=fixed_orientation_joints,
        scapula_r_axes=scapula_r_axes,
        scapula_l_axes=scapula_l_axes,
        spine_axes=spine_axes,
    )
    G = _propagate_transforms(xp, G_local, parents)

    # Shape blend shapes (simplified mesh for output)
    v_shaped = v_template + xp.einsum("vdi,bi->bvd", shapedirs, shape)

    # Pose blend shapes (SMPL-compatible)
    eye3 = xp.eye(3, dtype=dtype)
    R_smpl = xp.broadcast_to(eye3, (B, num_joints_smpl, 3, 3))
    # Set SKEL rotations into SMPL positions (copy=True handles broadcast->contiguous)
    idx_smpl = (slice(None), SMPL_JOINT_MAP)
    R_smpl = common.set(R_smpl, idx_smpl, G_local[:, :, :3, :3], copy=True, xp=xp)
    pose_feat = (R_smpl[:, 1:] - eye3).reshape(B, -1)
    pose_offsets = (pose_feat @ posedirs).reshape(B, Nv, 3)
    v_posed = v_shaped + pose_offsets

    # Skin LBS (optimized: separate R and t, avoid homogeneous coordinates)
    R_joint = G[:, :, :3, :3]  # [B, J, 3, 3]
    t_world = G[:, :, :3, 3]  # [B, J, 3]
    t_skin = t_world - xp.squeeze(R_joint @ J[..., None], axis=-1)  # [B, J, 3]

    W_R = xp.einsum("vj,bjkl->bvkl", skin_weights, R_joint)  # [B, V, 3, 3]
    W_t = xp.einsum("vj,bjk->bvk", skin_weights, t_skin)  # [B, V, 3]
    v_out = xp.squeeze(W_R @ v_posed[..., None], axis=-1) + W_t

    # Apply global transform
    v_out = v_out + global_translation[:, None]
    if global_rotation is not None:
        R = SO3.to_matrix(SO3.from_axis_angle(global_rotation, xp=xp), xp=xp)
        v_out = (R @ v_out.mT).mT

    return v_out + feet_offset


def forward_skeleton(
    # Model data
    v_template_full: Float[Array, "V_full 3"],
    shapedirs_full: Float[Array, "V_full 3 B"],
    J_regressor: Float[Array, "24 V_full"],
    parents: Int[Array, "23"],
    all_axes: Float[Array, "47 3"],
    rotation_indices: Int[Array, "24 3"],
    apose_R: Float[Array, "24 3 3"],
    apose_t: Float[Array, "24 3"],
    per_joint_rot: Float[Array, "24 3 3"],
    child: Int[Array, "24"],
    fixed_orientation_joints: Int[Array, "6"],
    feet_offset: Float[Array, "3"],
    scapula_r_axes: Float[Array, "3 3"],
    scapula_l_axes: Float[Array, "3 3"],
    spine_axes: Float[Array, "3 3"],
    # Inputs
    shape: Float[Array, "B 10"],
    pose: Float[Array, "B 46"],
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    xp: Any = None,
) -> Float[Array, "B 24 4 4"]:
    """Compute skeleton joint transforms [B, 24, 4, 4]."""
    assert shape.ndim == 2 and shape.shape[1] >= 1
    assert pose.ndim == 2 and pose.shape[1] == NUM_POSE_PARAMS
    assert global_rotation is None or (global_rotation.ndim == 2 and global_rotation.shape[1] == 3)
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(pose)
    B = pose.shape[0]
    dtype = pose.dtype

    if global_translation is None:
        global_translation = xp.zeros((B, 3), dtype=dtype)
    if shape.shape[0] == 1 and B > 1:
        shape = xp.broadcast_to(shape, (B, shape.shape[1]))

    # Shape blend shapes -> joint positions (use full-resolution for accurate skeleton)
    v_shaped_full = v_template_full + xp.einsum("vdi,bi->bvd", shapedirs_full, shape)
    J = xp.einsum("bvd,jv->bjd", v_shaped_full, J_regressor)
    J_rel = _compute_J_rel(xp, J, parents)

    # Forward kinematics
    G_local = _compute_local_transforms(
        xp=xp,
        pose=pose,
        J=J,
        J_rel=J_rel,
        all_axes=all_axes,
        rotation_indices=rotation_indices,
        apose_R=apose_R,
        apose_t=apose_t,
        per_joint_rot=per_joint_rot,
        child=child,
        fixed_orientation_joints=fixed_orientation_joints,
        scapula_r_axes=scapula_r_axes,
        scapula_l_axes=scapula_l_axes,
        spine_axes=spine_axes,
    )
    G = _propagate_transforms(xp, G_local, parents)

    # Apply global transform
    rot = G[:, :, :3, :3]
    trans = G[:, :, :3, 3]
    if global_rotation is not None:
        R = SO3.to_matrix(SO3.from_axis_angle(global_rotation, xp=xp), xp=xp)
        rot = R[:, None] @ rot
        trans = (R @ trans.mT).mT
    trans = trans + global_translation[:, None]

    # Add feet offset
    trans = trans + feet_offset

    # Build output transform
    last_row = xp.broadcast_to(xp.asarray([0, 0, 0, 1], dtype=dtype), (B, NUM_JOINTS, 1, 4))
    G = xp.concat([xp.concat([rot, trans[..., None]], axis=-1), last_row], axis=-2)
    return G


def _compute_J_rel(
    xp,
    J: Float[Array, "B 24 3"],
    parents: Int[Array, "23"],
) -> Float[Array, "B 24 3"]:
    """Compute relative joint positions."""
    J0 = J[:, :1]
    J_rest = J[:, 1:] - J[:, parents]
    return xp.concat([J0, J_rest], axis=1)


def _compute_local_transforms(
    xp,
    pose: Float[Array, "B 46"],
    J: Float[Array, "B 24 3"],
    J_rel: Float[Array, "B 24 3"],
    all_axes: Float[Array, "47 3"],
    rotation_indices: Int[Array, "24 3"],
    apose_R: Float[Array, "24 3 3"],
    apose_t: Float[Array, "24 3"],
    per_joint_rot: Float[Array, "24 3 3"],
    child: Int[Array, "24"],
    fixed_orientation_joints: Int[Array, "6"],
    scapula_r_axes: Float[Array, "3 3"],
    scapula_l_axes: Float[Array, "3 3"],
    spine_axes: Float[Array, "3 3"],
) -> Float[Array, "B 24 4 4"]:
    """Compute local joint transforms from pose parameters."""
    B = pose.shape[0]
    dtype = pose.dtype

    # Bone orientation correction
    Rk = _compute_bone_orientation(
        xp=xp,
        J_rel=J_rel,
        apose_t=apose_t,
        per_joint_rot=per_joint_rot,
        child=child,
        fixed_orientation_joints=fixed_orientation_joints,
    )
    Ra = xp.broadcast_to(apose_R[None], (B, NUM_JOINTS, 3, 3))

    # Batched joint rotations: convert all axis-angles to matrices at once
    # Pad pose with zero for identity rotation (used by joints with < 3 DOFs)
    zero_pad = xp.zeros((B, 1), dtype=dtype)
    pose_padded = xp.concat([pose, zero_pad], axis=1)
    axis_angles = pose_padded[..., None] * all_axes  # [B, 47, 3]
    all_R = SO3.to_matrix(SO3.from_axis_angle(axis_angles, xp=xp), xp=xp)  # [B, 47, 3, 3]

    # Compose rotations: Rp = R2 @ R1 @ R0 (identity-padded for joints with fewer DOFs)
    R0 = all_R[:, rotation_indices[:, 0]]  # [B, J, 3, 3]
    R1 = all_R[:, rotation_indices[:, 1]]
    R2 = all_R[:, rotation_indices[:, 2]]
    Rp = R2 @ (R1 @ R0)

    # Compose rotations: R = Rk @ Ra.T @ Rp @ Ra @ Rk.T
    Ra_T = Ra.mT
    Rk_T = Rk.mT
    R = Rk @ (Ra_T @ (Rp @ (Ra @ Rk_T)))

    # Translation with anatomical adjustments
    t_base = J_rel[..., None]  # [B, 24, 3, 1]

    # Compute offsets for special joints
    thorax_w = xp.linalg.vector_norm(J[:, 19] - J[:, 14], axis=1)
    thorax_h = xp.linalg.vector_norm(J[:, 12] - J[:, 11], axis=1)

    # Scapula offsets
    offset_r = _scapula_offset(xp, pose[:, 26], pose[:, 27], thorax_w, thorax_h, scapula_r_axes, is_left=False)
    offset_l = _scapula_offset(xp, pose[:, 36], pose[:, 37], thorax_w, thorax_h, scapula_l_axes, is_left=True)

    # Spine offsets
    offset_11 = _spine_offset(xp, pose[:, 17], pose[:, 18], xp.abs(J[:, 11, 1] - J[:, 0, 1]), spine_axes)
    offset_12 = _spine_offset(xp, pose[:, 20], pose[:, 21], xp.abs(J[:, 12, 1] - J[:, 11, 1]), spine_axes)
    offset_13 = _spine_offset(xp, pose[:, 23], pose[:, 24], xp.abs(J[:, 13, 1] - J[:, 12, 1]), spine_axes)

    # Build offset tensor
    zero = xp.zeros((B, 3, 1), dtype=dtype)
    offsets = [zero for _ in range(NUM_JOINTS)]
    offsets[14] = offset_r[:, :, None]
    offsets[19] = offset_l[:, :, None]
    offsets[11] = offset_11[:, :, None]
    offsets[12] = offset_12[:, :, None]
    offsets[13] = offset_13[:, :, None]
    offsets_tensor = xp.stack(offsets, axis=1)  # [B, 24, 3, 1]

    t = t_base + offsets_tensor

    return _homog_matrix(xp, R, t)


def _compute_bone_orientation(
    xp,
    J_rel: Float[Array, "B 24 3"],
    apose_t: Float[Array, "24 3"],
    per_joint_rot: Float[Array, "24 3 3"],
    child: Int[Array, "24"],
    fixed_orientation_joints: Int[Array, "6"],
) -> Float[Array, "B 24 3 3"]:
    """Compute per-joint orientation corrections."""
    B = J_rel.shape[0]
    dtype = J_rel.dtype

    bone_vec = J_rel[:, child]  # [B, 24, 3]

    # Special handling for certain joints
    bone_vec_16 = bone_vec[:, 16] + bone_vec[:, 17]
    bone_vec_21 = bone_vec[:, 21] + bone_vec[:, 22]
    bone_vec_12 = bone_vec[:, 11]

    # Build corrected bone_vec
    bone_vec_list = [bone_vec[:, i] for i in range(NUM_JOINTS)]
    bone_vec_list[16] = bone_vec_16
    bone_vec_list[21] = bone_vec_21
    bone_vec_list[12] = bone_vec_12
    bone_vec = xp.stack(bone_vec_list, axis=1)

    apose_vec = apose_t[child]  # [24, 3]
    apose_vec = xp.broadcast_to(apose_vec[None], (B, NUM_JOINTS, 3))

    # Special handling
    apose_vec_16 = apose_vec[:, 16] + apose_vec[:, 17]
    apose_vec_21 = apose_vec[:, 21] + apose_vec[:, 22]
    apose_vec_list = [apose_vec[:, i] for i in range(NUM_JOINTS)]
    apose_vec_list[16] = apose_vec_16
    apose_vec_list[21] = apose_vec_21
    apose_vec = xp.stack(apose_vec_list, axis=1)

    Gk_learned = xp.broadcast_to(per_joint_rot[None], (B, NUM_JOINTS, 3, 3))
    apose_corrected = xp.squeeze(Gk_learned @ apose_vec[..., None], axis=-1)

    Gk = _rotation_between_vectors(xp, apose_corrected, bone_vec)

    # Replace NaN values with zeros
    Gk = xp.where(xp.isnan(Gk), xp.zeros_like(Gk), Gk)

    # Set identity for fixed orientation joints
    eye3 = xp.eye(3, dtype=dtype)
    fixed = xp.broadcast_to(eye3, (B, NUM_JOINTS, 3, 3))
    mask = xp.zeros(NUM_JOINTS, dtype=xp.bool)
    mask = common.set(mask, (fixed_orientation_joints,), xp.asarray(True), xp=xp)
    mask = xp.broadcast_to(mask[None, :, None, None], Gk.shape)
    Gk = xp.where(mask, fixed, Gk)

    return Gk @ Gk_learned


def _scapula_offset(
    xp,
    abd: Float[Array, "B"],
    elev: Float[Array, "B"],
    thorax_w: Float[Array, "B"],
    thorax_h: Float[Array, "B"],
    axes: Float[Array, "3 3"],
    is_left: bool,
) -> Float[Array, "B 3"]:
    """Compute scapula joint offset."""
    pi = math.pi

    def pos(a, e, flip):
        if flip:
            a, e = -a, -e
        rx = thorax_w / 4 * xp.cos(e - pi / 4)
        sign = 1.0 if flip else -1.0
        return xp.stack(
            [
                sign * rx * xp.cos(a),
                -thorax_h / 2 * xp.sin(e - pi / 4),
                thorax_w / 4 * xp.sin(a),
            ],
            axis=1,
        )

    zero = xp.zeros_like(abd)
    return pos(abd, elev, is_left) - pos(zero, zero, is_left)


def _spine_offset(
    xp,
    yaw: Float[Array, "B"],
    pitch: Float[Array, "B"],
    height: Float[Array, "B"],
    axes: Float[Array, "3 3"],
) -> Float[Array, "B 3"]:
    """Compute spine joint offset."""

    def arc(angle, t, length):
        theta = angle * t
        # sinc(x) = sin(pi*x) / (pi*x), but numpy uses sinc(x) = sin(pi*x) / (pi*x)
        # We need sin(theta) / theta which is sinc(theta/pi) in numpy terms
        y = (
            length
            * t
            * xp.where(
                xp.abs(theta) < 1e-8,
                xp.ones_like(theta),
                xp.sin(theta) / theta,
            )
        )
        # For the second term: sinc(theta/(2*pi))^2 = (sin(theta/2) / (theta/2))^2
        half_theta = theta / 2
        sinc_half = xp.where(
            xp.abs(half_theta) < 1e-8,
            xp.ones_like(half_theta),
            xp.sin(half_theta) / half_theta,
        )
        x = 0.5 * length * angle * t**2 * sinc_half**2
        return x, y

    t = xp.ones_like(yaw)
    x1, y1 = arc(yaw, t, height)
    x2, y2 = arc(pitch, t, height)

    zero = xp.zeros_like(yaw)
    x1_0, y1_0 = arc(zero, t, height)
    x2_0, y2_0 = arc(zero, t, height)

    dx = xp.stack([-x1 + x1_0, y1 - y1_0 + y2 - y2_0, -x2 + x2_0], axis=1)
    return dx


def _propagate_transforms(
    xp,
    G_local: Float[Array, "B 24 4 4"],
    parents: Int[Array, "23"],
) -> Float[Array, "B 24 4 4"]:
    """Propagate local transforms to world space."""
    parent_list = parents.tolist()  # Works for numpy, torch, and jax arrays
    G_list = [G_local[:, 0]]
    for i in range(1, NUM_JOINTS):
        G_list.append(G_list[parent_list[i - 1]] @ G_local[:, i])
    return xp.stack(G_list, axis=1)


def _homog_matrix(
    xp,
    R: Float[Array, "B J 3 3"],
    t: Float[Array, "B J 3 1"],
) -> Float[Array, "B J 4 4"]:
    """Build [B, J, 4, 4] homogeneous matrix from rotation and translation."""
    B, J = R.shape[:2]
    dtype = R.dtype
    pad = xp.broadcast_to(xp.asarray([0, 0, 0, 1], dtype=dtype), (B, J, 1, 4))
    return xp.concat([xp.concat([R, t], axis=-1), pad], axis=-2)


def _skew(xp, v: Float[Array, "B N 3"]) -> Float[Array, "B N 3 3"]:
    """Skew-symmetric matrix from vector: [B, N, 3] -> [B, N, 3, 3]."""
    z = xp.zeros_like(v[..., :1])
    row0 = xp.concat([z, -v[..., 2:3], v[..., 1:2]], axis=-1)
    row1 = xp.concat([v[..., 2:3], z, -v[..., 0:1]], axis=-1)
    row2 = xp.concat([-v[..., 1:2], v[..., 0:1], z], axis=-1)
    return xp.stack([row0, row1, row2], axis=-2)


def _rotation_between_vectors(
    xp,
    a: Float[Array, "B N 3"],
    b: Float[Array, "B N 3"],
) -> Float[Array, "B N 3 3"]:
    """Rotation matrix that rotates normalized vectors a to b."""
    a_norm = xp.linalg.vector_norm(a, axis=-1, keepdims=True)
    b_norm = xp.linalg.vector_norm(b, axis=-1, keepdims=True)
    a = a / xp.where(a_norm > 1e-8, a_norm, xp.ones_like(a_norm))
    b = b / xp.where(b_norm > 1e-8, b_norm, xp.ones_like(b_norm))

    v = xp.linalg.cross(a, b)
    c = xp.sum(a * b, axis=-1)
    s = xp.linalg.vector_norm(v, axis=-1) + 1e-7

    K = _skew(xp, v)
    eye3 = xp.eye(3, dtype=a.dtype)
    I = xp.broadcast_to(eye3, (*a.shape[:-1], 3, 3))
    scale = ((1 - c) / (s**2))[..., None, None]
    return I + K + (K @ K) * scale


def from_native_args(
    shape: Float[Array, "B 10"],
    body_pose: Float[Array, "B 46"],
    root_rotation: Float[Array, "B 3"] | None = None,
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
) -> dict[str, Any]:
    """Convert native SKEL args to forward_* kwargs.

    Args:
        shape: Body shape parameters.
        body_pose: Body pose (46 params). First 3 are root rotation if root_rotation not given.
        root_rotation: Optional root rotation to override body_pose[:, :3].
        global_rotation: Optional additional global rotation (applied after skinning).
        global_translation: Optional global translation.
    """
    xp = get_namespace(body_pose)
    pose = body_pose
    if root_rotation is not None:
        pose = xp.concat([root_rotation, body_pose[:, 3:]], axis=1)

    return {
        "shape": shape,
        "pose": pose,
        "global_rotation": global_rotation,
        "global_translation": global_translation,
    }


def to_native_outputs(
    vertices: Float[Array, "B V 3"],
    transforms: Float[Array, "B J 4 4"],
    feet_offset: Float[Array, "3"],
) -> dict[str, Any]:
    """Convert forward_* outputs to native SKEL format.

    Native format returns joint positions (not transforms) without feet offset.
    """
    return {
        "vertices": vertices - feet_offset,
        "joints": transforms[..., :3, 3] - feet_offset,
    }
