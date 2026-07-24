"""SMPL deformation computations."""

from typing import Any

from jaxtyping import Float

from body_models.common import deformation, kinematics, skinning
from nanomanifold import SO3

from body_models.rotations import RotationType, rotation_ndim

Array = Any  # Generic array type (numpy, torch, jax)
Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).


SmplSkeletonIdentity = deformation.SkeletonIdentity
SmplIdentity = deformation.LinearIdentity
SmplPreparedPose = deformation.SkinningPose


def prepare_pose(
    # Model data
    posedirs: Float[Array, "P V*3"],
    kinematic_fronts: list[Front],
    body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    local_joint_offsets: Float[Array, "*batch J 3"],
    rest_joints: Float[Array, "*batch J 3"],
    xp: Any,
) -> SmplPreparedPose:
    """Precompute pose-dependent SMPL state for repeated forward passes."""
    num_rot_dims = rotation_ndim(rotation_type)
    pose_ndim = num_rot_dims + 1
    batch_shape = tuple(body_pose.shape[:-pose_ndim])
    local_joint_offsets = xp.broadcast_to(local_joint_offsets, (*batch_shape, *local_joint_offsets.shape[-2:]))
    rest_joints = xp.broadcast_to(rest_joints, (*batch_shape, *rest_joints.shape[-2:]))

    pose_matrices, T_world = _forward_core(
        xp=xp,
        kinematic_fronts=kinematic_fronts,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        rotation_type=rotation_type,
        local_joint_offsets=local_joint_offsets,
    )

    return {
        "skeleton_transforms": T_world,
        "skinning_transforms": skinning.bind_relative_transforms(T_world, rest_joints, xp=xp),
        "pose_offsets": deformation.pose_blend_shapes(pose_matrices, posedirs, xp=xp),
    }


def prepare_skeleton(
    kinematic_fronts: list[Front],
    body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*batch J 3"],
    xp: Any,
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed SMPL joint transforms."""
    batch_shape = body_pose.shape[: -(rotation_ndim(rotation_type) + 1)]
    local_joint_offsets = xp.broadcast_to(local_joint_offsets, (*batch_shape, *local_joint_offsets.shape[-2:]))
    _, transforms = _forward_core(
        xp=xp,
        kinematic_fronts=kinematic_fronts,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        rotation_type=rotation_type,
        local_joint_offsets=local_joint_offsets,
    )
    return transforms


def _forward_core(
    xp,
    kinematic_fronts: list[Front],
    body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    local_joint_offsets: Float[Array, "*batch J 3"],
) -> tuple[
    Float[Array, "*batch J 3 3"],
    Float[Array, "*batch J 4 4"],
]:
    """Core forward pass."""
    num_rot_dims = rotation_ndim(rotation_type)
    pose_ndim = num_rot_dims + 1
    batch_shape = tuple(body_pose.shape[:-pose_ndim])

    # Build full pose with pelvis rotation
    body_pose_matrices = SO3.convert(body_pose, src=rotation_type, dst="rotmat", xp=xp)
    if pelvis_rotation is None:
        pelvis_matrices = SO3.identity_as(
            body_pose_matrices,
            batch_dims=(*batch_shape, 1),
            rotation_type="rotmat",
            xp=xp,
        )
    else:
        pelvis_matrices = SO3.convert(
            pelvis_rotation,
            src=rotation_type,
            dst="rotmat",
            xp=xp,
        )[..., None, :, :]
    pose_matrices = xp.concat([pelvis_matrices, body_pose_matrices], axis=-3)

    T_world = kinematics.forward_kinematics(
        pose_matrices,
        local_joint_offsets,
        kinematic_fronts,
        xp=xp,
    )

    return pose_matrices, T_world


def prepare_identity(
    *,
    xp,
    v_template: Float[Array, "V 3"],
    shapedirs: Float[Array, "V D 10"],
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: list[int],
    shape: Float[Array, "*batch S"],
) -> SmplIdentity:
    """Precompute shape-dependent SMPL state for repeated forward passes."""
    shape_directions = shapedirs[:, :, : shape.shape[-1]]
    joint_directions = j_shapedirs[:, :, : shape.shape[-1]]
    return deformation.prepare_linear_identity(
        vertex_template=v_template,
        vertex_directions=shape_directions,
        joint_template=j_template,
        joint_directions=joint_directions,
        parents=parents,
        coefficients=shape,
        xp=xp,
    )


def prepare_skeleton_identity(
    *,
    xp: Any,
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: list[int],
    shape: Float[Array, "*batch S"],
) -> SmplSkeletonIdentity:
    """Prepare only shape-dependent SMPL joint state."""
    joint_directions = j_shapedirs[:, :, : shape.shape[-1]]
    return deformation.prepare_linear_skeleton(
        joint_template=j_template,
        joint_directions=joint_directions,
        parents=parents,
        coefficients=shape,
        xp=xp,
    )


__all__ = [
    "SmplIdentity",
    "SmplPreparedPose",
    "prepare_identity",
    "prepare_pose",
]
