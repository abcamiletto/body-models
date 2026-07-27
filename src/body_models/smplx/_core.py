"""Backend-independent SMPL-X pose and identity preparation."""

from typing import Any

from jaxtyping import Float
from nanomanifold import SO3

from body_models._common import deformation, kinematics, skinning
from body_models._rotations import RotationType, rotation_ndim

Array = Any
Front = tuple[list[int], list[int]]


SmplxSkeletonIdentity = deformation.SkeletonIdentity
SmplxIdentity = deformation.LinearIdentity
SmplxPreparedPose = deformation.SkinningPose


def prepare_pose(
    posedirs: Float[Array, "P V*3"],
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*batch J 3"],
    rest_joints: Float[Array, "*batch J 3"],
    xp: Any,
) -> SmplxPreparedPose:
    """Prepare SMPL-X transforms and pose-dependent vertex offsets."""
    num_rot_dims = rotation_ndim(rotation_type)
    pose_ndim = num_rot_dims + 1
    batch_shape = tuple(body_pose.shape[:-pose_ndim])
    if tuple(hand_pose.shape[:-pose_ndim]) != batch_shape or tuple(head_pose.shape[:-pose_ndim]) != batch_shape:
        raise ValueError("body_pose, hand_pose, and head_pose must have the same batch shape")
    local_joint_offsets = xp.broadcast_to(local_joint_offsets, (*batch_shape, *local_joint_offsets.shape[-2:]))
    rest_joints = xp.broadcast_to(rest_joints, (*batch_shape, *rest_joints.shape[-2:]))

    pose_matrices, world_transforms = _forward_core(
        xp=xp,
        kinematic_fronts=kinematic_fronts,
        hand_mean=hand_mean,
        body_pose=body_pose,
        hand_pose=hand_pose,
        head_pose=head_pose,
        pelvis_rotation=pelvis_rotation,
        rotation_type=rotation_type,
        local_joint_offsets=local_joint_offsets,
    )
    return {
        "skeleton_transforms": world_transforms,
        "skinning_transforms": skinning.bind_relative_transforms(world_transforms, rest_joints, xp=xp),
        "pose_offsets": deformation.pose_blend_shapes(pose_matrices, posedirs, xp=xp),
    }


def prepare_skeleton(
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*batch J 3"],
    xp: Any,
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed SMPL-X joint transforms."""
    pose_ndim = rotation_ndim(rotation_type) + 1
    batch_shape = tuple(body_pose.shape[:-pose_ndim])
    if tuple(hand_pose.shape[:-pose_ndim]) != batch_shape or tuple(head_pose.shape[:-pose_ndim]) != batch_shape:
        raise ValueError("body_pose, hand_pose, and head_pose must have the same batch shape")
    local_joint_offsets = xp.broadcast_to(local_joint_offsets, (*batch_shape, *local_joint_offsets.shape[-2:]))
    _, transforms = _forward_core(
        xp=xp,
        kinematic_fronts=kinematic_fronts,
        hand_mean=hand_mean,
        body_pose=body_pose,
        hand_pose=hand_pose,
        head_pose=head_pose,
        pelvis_rotation=pelvis_rotation,
        rotation_type=rotation_type,
        local_joint_offsets=local_joint_offsets,
    )
    return transforms


def _forward_core(
    xp: Any,
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    local_joint_offsets: Float[Array, "*batch J 3"],
) -> tuple[Float[Array, "*batch J 3 3"], Float[Array, "*batch J 4 4"]]:
    """Compose SMPL-X local rotations and run forward kinematics."""
    num_rot_dims = rotation_ndim(rotation_type)
    batch_shape = body_pose.shape[: -(num_rot_dims + 1)]

    if rotation_type == "axis_angle":
        hand_axis_angle = hand_pose
    else:
        hand_axis_angle = SO3.convert(hand_pose, src=rotation_type, dst="axis_angle", xp=xp)
    left_hand = hand_axis_angle[..., :15, :] + hand_mean[0].reshape(15, 3)
    right_hand = hand_axis_angle[..., 15:, :] + hand_mean[1].reshape(15, 3)
    hand_axis_angle = xp.concat([left_hand, right_hand], axis=-2)

    if pelvis_rotation is None:
        pelvis_matrices = SO3.identity_as(
            body_pose,
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
    body_matrices = SO3.convert(body_pose, src=rotation_type, dst="rotmat", xp=xp)
    head_matrices = SO3.convert(head_pose, src=rotation_type, dst="rotmat", xp=xp)
    hand_matrices = SO3.convert(hand_axis_angle, src="axis_angle", dst="rotmat", xp=xp)
    pose_matrices = xp.concat([pelvis_matrices, body_matrices, head_matrices, hand_matrices], axis=-3)
    world_transforms = kinematics.forward_kinematics(
        pose_matrices,
        local_joint_offsets,
        kinematic_fronts,
        xp=xp,
    )
    return pose_matrices, world_transforms


def prepare_identity(
    *,
    xp: Any,
    v_template: Float[Array, "V 3"],
    shapedirs: Float[Array, "V 3 S"],
    exprdirs: Float[Array, "V 3 E"],
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    j_exprdirs: Float[Array, "J 3 E"],
    parents: list[int],
    shape: Float[Array, "*batch S"],
    expression: Float[Array, "*batch E"],
) -> SmplxIdentity:
    """Prepare shape- and expression-dependent SMPL-X state."""
    _validate_identity_parameters(shape, expression)
    shape_dim = shape.shape[-1]
    expression_dim = expression.shape[-1]
    parameters = xp.concat([shape, expression], axis=-1)
    vertex_directions = xp.concat(
        [shapedirs[:, :, :shape_dim], exprdirs[:, :, :expression_dim]],
        axis=-1,
    )
    joint_directions = xp.concat(
        [j_shapedirs[:, :, :shape_dim], j_exprdirs[:, :, :expression_dim]],
        axis=-1,
    )
    return deformation.prepare_linear_identity(
        vertex_template=v_template,
        vertex_directions=vertex_directions,
        joint_template=j_template,
        joint_directions=joint_directions,
        parents=parents,
        coefficients=parameters,
        xp=xp,
    )


def prepare_skeleton_identity(
    *,
    xp: Any,
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    j_exprdirs: Float[Array, "J 3 E"],
    parents: list[int],
    shape: Float[Array, "*batch S"],
    expression: Float[Array, "*batch E"],
) -> SmplxSkeletonIdentity:
    """Prepare only identity-dependent SMPL-X joint state."""
    _validate_identity_parameters(shape, expression)
    shape_dim = shape.shape[-1]
    expression_dim = expression.shape[-1]
    parameters = xp.concat([shape, expression], axis=-1)
    joint_directions = xp.concat(
        [j_shapedirs[:, :, :shape_dim], j_exprdirs[:, :, :expression_dim]],
        axis=-1,
    )
    return deformation.prepare_linear_skeleton(
        joint_template=j_template,
        joint_directions=joint_directions,
        parents=parents,
        coefficients=parameters,
        xp=xp,
    )


def _validate_identity_parameters(
    shape: Float[Array, "*batch S"],
    expression: Float[Array, "*batch E"],
) -> None:
    if shape.ndim < 1 or shape.shape[-1] < 1:
        raise ValueError("shape must have shape [..., S] with S >= 1")
    if expression.ndim < 1 or expression.shape[-1] < 1:
        raise ValueError("expression must have shape [..., E] with E >= 1")


__all__ = ["SmplxIdentity", "SmplxPreparedPose", "prepare_identity", "prepare_pose"]
