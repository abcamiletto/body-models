"""MHR deformation computations."""

import math
from typing import Any

from jaxtyping import Float
from nanomanifold import SO3

from body_models import _common as common
from body_models._common import deformation
from body_models._runtime import ArrayRuntime

Array = Any  # Generic array type (numpy, torch, jax)

_LN2 = math.log(2)


def _pose_coefficients(
    joint_params: Float[Array, "*batch J 7"],
    hidden_weights: Float[Array, "input hidden"],
    *,
    xp: Any,
) -> Float[Array, "*batch hidden"]:
    dtype = joint_params.dtype

    euler = joint_params[..., 2:, 3:6]
    rot = SO3.conversions.from_euler_to_rotmat(euler, convention="xyz", xp=xp)
    feat = xp.concat([rot[..., 0], rot[..., 1]], axis=-1)
    feat = common.at_set(feat, (..., 0), feat[..., 0] - 1.0, copy=False, xp=xp)
    feat = common.at_set(feat, (..., 4), feat[..., 4] - 1.0, copy=False, xp=xp)

    batch_shape = feat.shape[:-2]
    feat_flat = feat.reshape(*batch_shape, -1)
    h = feat_flat @ hidden_weights
    return xp.maximum(h, xp.asarray(0.0, dtype=dtype))


def prepare_pose(
    runtime: ArrayRuntime,
    joint_offsets: Float[Array, "J 3"],
    joint_pre_rotations: Float[Array, "J 4"],
    parameter_transform: Float[Array, "D N"],
    tree: common.KinematicTree,
    num_joints: int,
    shape_dim: int,
    bind_inv_linear: Float[Array, "J 3 3"],
    bind_inv_translation: Float[Array, "J 3"],
    corrective_hidden_weights: Float[Array, "input hidden"],
    pose: Float[Array, "B 204"],
) -> deformation.SkinningPose:
    """Precompute pose-dependent MHR state for repeated forward passes."""
    if pose.ndim < 1 or pose.shape[-1] != 204:
        raise ValueError(f"pose must have shape [..., 204], got {tuple(pose.shape)}")
    world, j_p = _forward_skeleton_core(
        runtime=runtime,
        pose=pose,
        joint_offsets=joint_offsets,
        joint_pre_rotations=joint_pre_rotations,
        parameter_transform=parameter_transform,
        tree=tree,
        num_joints=num_joints,
        shape_dim=shape_dim,
    )
    xp = runtime.xp
    return {
        "skeleton_transforms": _scale_transform_translations(xp, world),
        "skinning_transforms": _skinning_transforms(
            xp,
            world_transforms=world,
            bind_inv_linear=bind_inv_linear,
            bind_inv_translation=bind_inv_translation,
        ),
        "pose_coefficients": _pose_coefficients(
            j_p,
            corrective_hidden_weights,
            xp=xp,
        ),
    }


def prepare_skeleton(
    runtime: ArrayRuntime,
    joint_offsets: Float[Array, "J 3"],
    joint_pre_rotations: Float[Array, "J 4"],
    parameter_transform: Float[Array, "D N"],
    tree: common.KinematicTree,
    num_joints: int,
    shape_dim: int,
    pose: Float[Array, "B 204"],
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed MHR joint transforms."""
    world, _ = _forward_skeleton_core(
        runtime=runtime,
        pose=pose,
        joint_offsets=joint_offsets,
        joint_pre_rotations=joint_pre_rotations,
        parameter_transform=parameter_transform,
        tree=tree,
        num_joints=num_joints,
        shape_dim=shape_dim,
    )
    return _scale_transform_translations(runtime.xp, world)


def prepare_identity(
    *,
    xp,
    base_vertices: Float[Array, "V 3"],
    blendshape_dirs: Float[Array, "117 V 3"],
    shape: Float[Array, "*batch 45"],
    expression: Float[Array, "*batch 72"],
) -> deformation.SkinningIdentity:
    """Precompute shape- and expression-dependent MHR state for repeated forward passes."""
    if shape.ndim < 1 or shape.shape[-1] != 45:
        raise ValueError(f"shape must have shape [..., 45], got {tuple(shape.shape)}")
    if expression.ndim < 1 or expression.shape[-1] != 72:
        raise ValueError(f"expression must have shape [..., 72], got {tuple(expression.shape)}")
    coeffs = xp.concat([shape, expression], axis=-1)
    return {
        "rest_vertices": (base_vertices + xp.einsum("...i,ivk->...vk", coeffs, blendshape_dirs)) * 0.01,
    }


def _skinning_transforms(
    xp,
    *,
    world_transforms: Float[Array, "*batch J 4 4"],
    bind_inv_linear: Float[Array, "J 3 3"],
    bind_inv_translation: Float[Array, "J 3"],
) -> Float[Array, "*batch J 4 4"]:
    lin_g = world_transforms[..., :3, :3]
    joint_translations = world_transforms[..., :3, 3]
    lin = xp.einsum("...jik,jkl->...jil", lin_g, bind_inv_linear)
    t = xp.einsum("...jik,jk->...ji", lin_g, bind_inv_translation) + joint_translations
    return _transforms_from_linear_translation(xp, lin, t * 0.01)


def _forward_skeleton_core(
    runtime: ArrayRuntime,
    pose: Float[Array, "B 204"],
    joint_offsets: Float[Array, "J 3"],
    joint_pre_rotations: Float[Array, "J 4"],
    parameter_transform: Float[Array, "D N"],
    tree: common.KinematicTree,
    num_joints: int,
    shape_dim: int,
) -> tuple[Float[Array, "B J 4 4"], Float[Array, "B J 7"]]:
    xp = runtime.xp
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

    local_scale = xp.exp(_LN2 * j_p[..., 6:7])
    local_rotation = SO3.conversions.from_quat_to_rotmat(q_l, convention="xyzw", xp=xp)
    local_transforms = common.affine_transforms(local_rotation * local_scale[..., None], t_l, xp=xp)
    return runtime._compose_kinematic_tree(local_transforms, tree), j_p


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


def _scale_transform_translations(xp, transforms: Float[Array, "B J 4 4"]) -> Float[Array, "B J 4 4"]:
    return common.affine_transforms(
        transforms[..., :3, :3],
        transforms[..., :3, 3] * 0.01,
        xp=xp,
    )


def _transforms_from_linear_translation(
    xp,
    linear: Float[Array, "B J 3 3"],
    translation: Float[Array, "B J 3"],
) -> Float[Array, "B J 4 4"]:
    return common.affine_transforms(linear, translation, xp=xp)
