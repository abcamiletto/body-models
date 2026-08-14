"""Backend-independent SKEL identity and pose preparation."""

import math
from collections.abc import Sequence
from typing import Any, TypedDict

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import _common as common
from body_models._common import skinning
from body_models._runtime import ArrayRuntime

Array = Any  # Generic array type (numpy, torch, jax)

# SKEL uses SMPL-compatible pose blend shapes - this maps SKEL joints to SMPL joints
SMPL_JOINT_MAP = [0, 2, 5, 8, 8, 11, 1, 4, 7, 7, 10, 3, 6, 15, 14, 17, 19, 0, 21, 13, 16, 18, 0, 20]

# Constants
NUM_JOINTS = 24
NUM_POSE_PARAMS = 46
NUM_SHAPE_COEFFS = 10


class SkelSkeletonIdentity(TypedDict):
    """Shape-dependent joint state needed to pose the SKEL skeleton."""

    rest_joints: Float[Array, "*batch 24 3"]
    local_joint_offsets: Float[Array, "*batch 24 3"]


class SkelIdentity(SkelSkeletonIdentity):
    """Complete shape-dependent SKEL mesh state."""

    rest_vertices: Float[Array, "*batch V 3"]


def prepare_pose(
    runtime: ArrayRuntime,
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
    tree: common.KinematicTree,
    num_joints_smpl: int,
    pose: Float[Array, "*batch 46"],
    *,
    local_joint_offsets: Float[Array, "*batch 24 3"],
    rest_joints: Float[Array, "*batch 24 3"],
) -> common.deformation.SkinningPose:
    """Precompute pose-dependent SKEL state for repeated forward passes."""
    if pose.ndim < 1 or pose.shape[-1] != NUM_POSE_PARAMS:
        raise ValueError(f"pose must have shape [..., {NUM_POSE_PARAMS}]")
    xp = runtime.xp
    batch_shape = tuple(pose.shape[:-1])
    G_local = _compute_local_transforms(
        xp=xp,
        pose=pose,
        J=rest_joints,
        J_rel=local_joint_offsets,
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
    G = runtime._compose_kinematic_tree(G_local, tree)

    eye3 = common.eye_as(G_local[..., :3, :3], batch_dims=(*batch_shape, 1), xp=xp)
    R_smpl = xp.broadcast_to(eye3, (*batch_shape, num_joints_smpl, 3, 3))
    R_smpl = common.at_set(
        R_smpl, (..., SMPL_JOINT_MAP, slice(None), slice(None)), G_local[..., :, :3, :3], copy=True, xp=xp
    )
    pose_feat = (R_smpl[..., 1:, :, :] - eye3).reshape(*batch_shape, -1)
    return {
        "skeleton_transforms": G,
        "skinning_transforms": skinning.bind_relative_transforms(G, rest_joints, xp=xp),
        "pose_coefficients": pose_feat,
    }


def prepare_skeleton(
    runtime: ArrayRuntime,
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
    tree: common.KinematicTree,
    pose: Float[Array, "*batch 46"],
    *,
    local_joint_offsets: Float[Array, "*batch 24 3"],
    rest_joints: Float[Array, "*batch 24 3"],
    joint_indices: Sequence[int] | None = None,
) -> Float[Array, "*batch 24 4 4"]:
    """Prepare only posed SKEL joint transforms."""
    if pose.ndim < 1 or pose.shape[-1] != NUM_POSE_PARAMS:
        raise ValueError(f"pose must have shape [..., {NUM_POSE_PARAMS}]")
    xp = runtime.xp
    local = _compute_local_transforms(
        xp=xp,
        pose=pose,
        J=rest_joints,
        J_rel=local_joint_offsets,
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
    selection = None
    if joint_indices is not None:
        selection = tree.select(joint_indices)
        joints = xp.asarray(selection.joints, dtype=xp.int32)
        local = local[..., joints, :, :]
        tree = selection.tree
    skeleton = runtime._compose_kinematic_tree(local, tree)
    if selection is None:
        return skeleton
    return skeleton[..., xp.asarray(selection.order, dtype=xp.int32), :, :]


def prepare_identity(
    v_template: Float[Array, "V 3"],
    shapedirs: Float[Array, "V 3 B"],
    j_template: Float[Array, "24 3"],
    j_shapedirs: Float[Array, "24 3 B"],
    parent: Int[Array, "23"],
    shape: Float[Array, "*batch 10"],
    *,
    xp: Any,
) -> SkelIdentity:
    """Precompute shape-dependent SKEL state for repeated forward passes."""
    identity = prepare_skeleton_identity(j_template, j_shapedirs, parent, shape, xp=xp)
    return {
        "rest_joints": identity["rest_joints"],
        "local_joint_offsets": identity["local_joint_offsets"],
        "rest_vertices": v_template + xp.einsum("vdi,...i->...vd", shapedirs, shape),
    }


def prepare_skeleton_identity(
    j_template: Float[Array, "24 3"],
    j_shapedirs: Float[Array, "24 3 B"],
    parent: Int[Array, "23"],
    shape: Float[Array, "*batch 10"],
    *,
    xp: Any,
) -> SkelSkeletonIdentity:
    """Prepare only shape-dependent SKEL joint state."""
    if shape.ndim < 1 or shape.shape[-1] != NUM_SHAPE_COEFFS:
        raise ValueError(f"shape must have shape [..., {NUM_SHAPE_COEFFS}]")
    joints = j_template + xp.einsum("jdi,...i->...jd", j_shapedirs, shape)
    return {
        "rest_joints": joints,
        "local_joint_offsets": _compute_J_rel(xp, joints, parent),
    }


def _compute_J_rel(
    xp,
    J: Float[Array, "B 24 3"],
    parents: Int[Array, "23"],
) -> Float[Array, "B 24 3"]:
    """Compute relative joint positions."""
    J0 = J[..., :1, :]
    J_rest = J[..., 1:, :] - J[..., parents, :]
    return xp.concat([J0, J_rest], axis=-2)


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
    batch_shape = pose.shape[:-1]

    # Bone orientation correction
    Rk = _compute_bone_orientation(
        xp=xp,
        J_rel=J_rel,
        apose_t=apose_t,
        per_joint_rot=per_joint_rot,
        child=child,
        fixed_orientation_joints=fixed_orientation_joints,
    )
    Ra = xp.broadcast_to(apose_R, (*batch_shape, NUM_JOINTS, 3, 3))

    # Batched joint rotations: convert all axis-angles to matrices at once
    # Pad pose with zero for identity rotation (used by joints with < 3 DOFs)
    zero_pad = common.zeros_as(pose, shape=(*batch_shape, 1), xp=xp)
    pose_padded = xp.concat([pose, zero_pad], axis=-1)
    axis_angles = pose_padded[..., None] * all_axes
    all_R = SO3.conversions.from_axis_angle_to_rotmat(axis_angles, xp=xp)  # [B, 47, 3, 3]

    # Compose rotations: Rp = R2 @ R1 @ R0 (identity-padded for joints with fewer DOFs)
    R0 = all_R[..., rotation_indices[:, 0], :, :]
    R1 = all_R[..., rotation_indices[:, 1], :, :]
    R2 = all_R[..., rotation_indices[:, 2], :, :]
    Rp = R2 @ (R1 @ R0)

    # Compose rotations: R = Rk @ Ra.T @ Rp @ Ra @ Rk.T
    Ra_T = Ra.mT
    Rk_T = Rk.mT
    R = Rk @ (Ra_T @ (Rp @ (Ra @ Rk_T)))

    # Translation with anatomical adjustments
    t_base = J_rel[..., None]  # [B, 24, 3, 1]

    # Compute offsets for special joints
    thorax_w = xp.linalg.vector_norm(J[..., 19, :] - J[..., 14, :], axis=-1)
    thorax_h = xp.linalg.vector_norm(J[..., 12, :] - J[..., 11, :], axis=-1)

    # Scapula offsets
    offset_r = _scapula_offset(xp, pose[..., 26], pose[..., 27], thorax_w, thorax_h, scapula_r_axes, is_left=False)
    offset_l = _scapula_offset(xp, pose[..., 36], pose[..., 37], thorax_w, thorax_h, scapula_l_axes, is_left=True)

    # Spine offsets
    offset_11 = _spine_offset(xp, pose[..., 17], pose[..., 18], xp.abs(J[..., 11, 1] - J[..., 0, 1]), spine_axes)
    offset_12 = _spine_offset(xp, pose[..., 20], pose[..., 21], xp.abs(J[..., 12, 1] - J[..., 11, 1]), spine_axes)
    offset_13 = _spine_offset(xp, pose[..., 23], pose[..., 24], xp.abs(J[..., 13, 1] - J[..., 12, 1]), spine_axes)

    # Build offset tensor
    zero = common.zeros_as(pose, shape=(*batch_shape, 3, 1), xp=xp)
    offsets = [zero for _ in range(NUM_JOINTS)]
    offsets[14] = offset_r[..., :, None]
    offsets[19] = offset_l[..., :, None]
    offsets[11] = offset_11[..., :, None]
    offsets[12] = offset_12[..., :, None]
    offsets[13] = offset_13[..., :, None]
    offsets_tensor = xp.stack(offsets, axis=-3)

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
    batch_shape = J_rel.shape[:-2]

    bone_vec = J_rel[..., child, :]

    # Special handling for certain joints
    bone_vec_16 = bone_vec[..., 16, :] + bone_vec[..., 17, :]
    bone_vec_21 = bone_vec[..., 21, :] + bone_vec[..., 22, :]
    bone_vec_12 = bone_vec[..., 11, :]

    # Build corrected bone_vec
    bone_vec_list = [bone_vec[..., i, :] for i in range(NUM_JOINTS)]
    bone_vec_list[16] = bone_vec_16
    bone_vec_list[21] = bone_vec_21
    bone_vec_list[12] = bone_vec_12
    bone_vec = xp.stack(bone_vec_list, axis=-2)

    apose_vec = apose_t[child]  # [24, 3]
    apose_vec = xp.broadcast_to(apose_vec, (*batch_shape, NUM_JOINTS, 3))

    # Special handling
    apose_vec_16 = apose_vec[..., 16, :] + apose_vec[..., 17, :]
    apose_vec_21 = apose_vec[..., 21, :] + apose_vec[..., 22, :]
    apose_vec_list = [apose_vec[..., i, :] for i in range(NUM_JOINTS)]
    apose_vec_list[16] = apose_vec_16
    apose_vec_list[21] = apose_vec_21
    apose_vec = xp.stack(apose_vec_list, axis=-2)

    Gk_learned = xp.broadcast_to(per_joint_rot, (*batch_shape, NUM_JOINTS, 3, 3))
    apose_corrected = xp.squeeze(Gk_learned @ apose_vec[..., None], axis=-1)

    Gk = common.rotation_between_vectors(apose_corrected, bone_vec, xp=xp)

    # Set identity for fixed orientation joints
    eye3 = common.eye_as(per_joint_rot, batch_dims=(*batch_shape, NUM_JOINTS), xp=xp)
    fixed = xp.broadcast_to(eye3, (*batch_shape, NUM_JOINTS, 3, 3))
    mask = common.zeros_as(fixed, shape=(NUM_JOINTS,), xp=xp)
    mask = xp.asarray(mask, dtype=xp.bool)
    mask = common.at_set(mask, (fixed_orientation_joints,), xp.asarray(True), xp=xp)
    mask = xp.broadcast_to(mask[..., None, None], Gk.shape)
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
    # Keep scalar arithmetic in the pose dtype for forward-mode autodiff.
    quarter_pi = xp.asarray(math.pi / 4, dtype=elev.dtype)

    def pos(a, e, flip):
        if flip:
            a, e = -a, -e
        tilt = e - quarter_pi
        rx = thorax_w / 4 * xp.cos(tilt)
        sign = 1 if flip else -1
        return xp.stack(
            [
                sign * rx * xp.cos(a),
                -thorax_h / 2 * xp.sin(tilt),
                thorax_w / 4 * xp.sin(a),
            ],
            axis=-1,
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
        small = xp.abs(theta) < 1e-8
        safe_theta = xp.where(small, xp.ones_like(theta), theta)
        y = length * t * xp.where(small, xp.ones_like(theta), xp.sin(safe_theta) / safe_theta)
        # For the second term: sinc(theta/(2*pi))^2 = (sin(theta/2) / (theta/2))^2
        half_theta = theta / 2
        small_half = xp.abs(half_theta) < 1e-8
        safe_half_theta = xp.where(small_half, xp.ones_like(half_theta), half_theta)
        sinc_half = xp.where(
            small_half,
            xp.ones_like(half_theta),
            xp.sin(safe_half_theta) / safe_half_theta,
        )
        x = 0.5 * length * angle * t**2 * sinc_half**2
        return x, y

    t = xp.ones_like(yaw)
    x1, y1 = arc(yaw, t, height)
    x2, y2 = arc(pitch, t, height)

    zero = xp.zeros_like(yaw)
    x1_0, y1_0 = arc(zero, t, height)
    x2_0, y2_0 = arc(zero, t, height)

    dx = xp.stack([-x1 + x1_0, y1 - y1_0 + y2 - y2_0, -x2 + x2_0], axis=-1)
    return dx


def _homog_matrix(
    xp,
    R: Float[Array, "B J 3 3"],
    t: Float[Array, "B J 3 1"],
) -> Float[Array, "B J 4 4"]:
    """Build [B, J, 4, 4] homogeneous matrix from rotation and translation."""
    batch_shape = R.shape[:-3]
    J = R.shape[-3]
    dtype = R.dtype
    pad = common.zeros_as(R, shape=(*batch_shape, J, 1, 4), xp=xp)
    pad = common.at_set(pad, (..., 0, 3), xp.asarray(1.0, dtype=dtype), xp=xp)
    return xp.concat([xp.concat([R, t], axis=-1), pad], axis=-2)


__all__ = ["SkelIdentity", "prepare_identity", "prepare_pose"]
