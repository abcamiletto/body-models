"""MHR deformation computations."""

import math
from typing import Any, TypedDict

from jaxtyping import Float
from nanomanifold import SO3

from body_models import common

Array = Any  # Generic array type (numpy, torch, jax)
Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).

_LN2 = math.log(2)


class MhrIdentity(TypedDict):
    """Complete shape- and expression-dependent MHR mesh state."""

    rest_vertices: Float[Array, "*batch V 3"]


class MhrPreparedPose(TypedDict):
    """Complete pose-dependent MHR mesh state."""

    skeleton_transforms: Float[Array, "*batch J 4 4"]
    skinning_transforms: Float[Array, "*batch J 4 4"]
    pose_offsets: Float[Array, "*batch V 3"]


def _apply_pose_correctives(
    joint_params: Float[Array, "B J 7"],
    W1: Float[Array, "3000 750"],
    W2: Float[Array, "V*3 3000"],
    *,
    xp: Any,
) -> Float[Array, "B V 3"]:
    V = W2.shape[0] // 3
    dtype = joint_params.dtype

    euler = joint_params[..., 2:, 3:6]
    rot = SO3.conversions.from_euler_to_rotmat(euler, convention="xyz", xp=xp)
    feat = xp.concat([rot[..., 0], rot[..., 1]], axis=-1)
    feat = common.set(feat, (..., 0), feat[..., 0] - 1.0, copy=False, xp=xp)
    feat = common.set(feat, (..., 4), feat[..., 4] - 1.0, copy=False, xp=xp)

    batch_shape = feat.shape[:-2]
    feat_flat = feat.reshape(*batch_shape, -1)
    h = feat_flat @ W1.T
    h = xp.maximum(h, xp.asarray(0.0, dtype=dtype))
    out = h @ W2.T

    return out.reshape(*batch_shape, V, 3)


def prepare_pose(
    joint_offsets: Float[Array, "J 3"],
    joint_pre_rotations: Float[Array, "J 4"],
    parameter_transform: Float[Array, "D N"],
    kinematic_fronts: list[Front],
    num_joints: int,
    shape_dim: int,
    bind_inv_linear: Float[Array, "J 3 3"],
    bind_inv_translation: Float[Array, "J 3"],
    corrective_W1: Float[Array, "3000 750"],
    corrective_W2: Float[Array, "V*3 3000"],
    pose: Float[Array, "B 204"],
    *,
    xp: Any,
) -> MhrPreparedPose:
    """Precompute pose-dependent MHR state for repeated forward passes."""
    assert pose.ndim >= 1 and pose.shape[-1] == 204
    t_g, r_g, s_g, j_p = _forward_skeleton_core(
        xp=xp,
        pose=pose,
        joint_offsets=joint_offsets,
        joint_pre_rotations=joint_pre_rotations,
        parameter_transform=parameter_transform,
        kinematic_fronts=kinematic_fronts,
        num_joints=num_joints,
        shape_dim=shape_dim,
    )
    return {
        "skeleton_transforms": _trs_to_transforms(xp, t_g * 0.01, r_g, s_g),
        "skinning_transforms": _skinning_transforms(
            xp,
            joint_translations=t_g,
            joint_rotations=r_g,
            joint_scales=s_g,
            bind_inv_linear=bind_inv_linear,
            bind_inv_translation=bind_inv_translation,
        ),
        "pose_offsets": _apply_pose_correctives(j_p, corrective_W1, corrective_W2, xp=xp) * 0.01,
    }


def prepare_skeleton(
    joint_offsets: Float[Array, "J 3"],
    joint_pre_rotations: Float[Array, "J 4"],
    parameter_transform: Float[Array, "D N"],
    kinematic_fronts: list[Front],
    num_joints: int,
    shape_dim: int,
    pose: Float[Array, "B 204"],
    *,
    xp: Any,
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed MHR joint transforms."""
    translations, rotations, scales, _ = _forward_skeleton_core(
        xp=xp,
        pose=pose,
        joint_offsets=joint_offsets,
        joint_pre_rotations=joint_pre_rotations,
        parameter_transform=parameter_transform,
        kinematic_fronts=kinematic_fronts,
        num_joints=num_joints,
        shape_dim=shape_dim,
    )
    return _trs_to_transforms(xp, translations * 0.01, rotations, scales)


def prepare_identity(
    *,
    xp,
    base_vertices: Float[Array, "V 3"],
    blendshape_dirs: Float[Array, "117 V 3"],
    shape: Float[Array, "*batch 45"],
    expression: Float[Array, "*batch 72"],
) -> MhrIdentity:
    """Precompute shape- and expression-dependent MHR state for repeated forward passes."""
    assert shape.ndim >= 1 and shape.shape[-1] >= 1
    coeffs = xp.concat([shape, expression], axis=-1)
    return {
        "rest_vertices": (base_vertices + xp.einsum("...i,ivk->...vk", coeffs, blendshape_dirs)) * 0.01,
    }


def _skinning_transforms(
    xp,
    *,
    joint_translations: Float[Array, "*batch J 3"],
    joint_rotations: Float[Array, "*batch J 3 3"],
    joint_scales: Float[Array, "*batch J 1"],
    bind_inv_linear: Float[Array, "J 3 3"],
    bind_inv_translation: Float[Array, "J 3"],
) -> Float[Array, "*batch J 4 4"]:
    lin_g = joint_rotations * joint_scales[..., None]
    lin = xp.einsum("...jik,jkl->...jil", lin_g, bind_inv_linear)
    t = xp.einsum("...jik,jk->...ji", lin_g, bind_inv_translation) + joint_translations
    return _transforms_from_linear_translation(xp, lin, t * 0.01)


def _forward_skeleton_core(
    xp,
    pose: Float[Array, "B 204"],
    joint_offsets: Float[Array, "J 3"],
    joint_pre_rotations: Float[Array, "J 4"],
    parameter_transform: Float[Array, "D N"],
    kinematic_fronts: list[Front],
    num_joints: int,
    shape_dim: int,
) -> tuple[Float[Array, "B J 3"], Float[Array, "B J 3 3"], Float[Array, "B J 1"], Float[Array, "B J 7"]]:
    j_p = _pose_to_joint_params(xp, pose, parameter_transform, num_joints, shape_dim)

    t_l = j_p[..., :3] + joint_offsets
    euler = j_p[..., 3:6]

    q_local = SO3.canonicalize(
        SO3.conversions.from_euler_to_quat(euler, euler_convention="xyz", quat_convention="xyzw", xp=xp),
        convention="xyzw",
        xp=xp,
    )
    q_l = SO3.canonicalize(
        SO3.multiply(joint_pre_rotations, q_local, convention="xyzw", xp=xp), convention="xyzw", xp=xp
    )

    s_l = xp.exp(_LN2 * j_p[..., 6:7])
    t_g, r_g, s_g = _compose_global_trs(xp, t_l, q_l, s_l, kinematic_fronts, num_joints)

    return t_g, r_g, s_g, j_p


def _pose_to_joint_params(
    xp,
    pose: Float[Array, "B 204"],
    parameter_transform: Float[Array, "D N"],
    num_joints: int,
    shape_dim: int,
) -> Float[Array, "B J 7"]:
    """Convert pose vector to per-joint parameters [B, J, 7]."""
    batch_shape = pose.shape[:-1]
    pad = common.zeros_as(pose, shape=(*batch_shape, shape_dim), xp=xp)
    j_p = xp.einsum("dn,...n->...d", parameter_transform, xp.concat([pose, pad], axis=-1))
    return j_p.reshape(*batch_shape, num_joints, 7)


def _compose_global_trs(
    xp,
    t_l: Float[Array, "B J 3"],
    q_l: Float[Array, "B J 4"],
    s_l: Float[Array, "B J 1"],
    kinematic_fronts: list[Front],
    num_joints: int,
) -> tuple[Float[Array, "B J 3"], Float[Array, "B J 3 3"], Float[Array, "B J 1"]]:
    r_l = SO3.conversions.from_quat_to_rotmat(q_l, convention="xyzw", xp=xp)

    t_results: list[Float[Array, "B 3"] | None] = [None] * num_joints
    s_results: list[Float[Array, "B 1"] | None] = [None] * num_joints
    r_results: list[Float[Array, "B 3 3"] | None] = [None] * num_joints

    for joints, parents in kinematic_fronts:
        if parents[0] < 0:
            for j in joints:
                t_results[j] = t_l[..., j, :]
                s_results[j] = s_l[..., j, :]
                r_results[j] = r_l[..., j, :, :]
        else:
            for j, p in zip(joints, parents):
                r_results[j] = r_results[p] @ r_l[..., j, :, :]
                s_results[j] = s_results[p] * s_l[..., j, :]
                r_ps = r_results[p] * s_results[p][..., :, None]
                t_ps = xp.squeeze(r_ps @ t_l[..., j, :, None], axis=-1)
                t_results[j] = t_ps + t_results[p]

    return (
        xp.stack(t_results, axis=-2),
        xp.stack(r_results, axis=-3),
        xp.stack(s_results, axis=-2),
    )


def _trs_to_transforms(
    xp,
    t: Float[Array, "B J 3"],
    r: Float[Array, "B J 3 3"],
    s: Float[Array, "B J 1"],
) -> Float[Array, "B J 4 4"]:
    R = r * s[..., None]
    return common.affine_transforms(R, t, xp=xp)


def _transforms_from_linear_translation(
    xp,
    linear: Float[Array, "B J 3 3"],
    translation: Float[Array, "B J 3"],
) -> Float[Array, "B J 4 4"]:
    return common.affine_transforms(linear, translation, xp=xp)
