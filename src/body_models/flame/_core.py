"""Backend-independent FLAME pose and identity preparation."""

from typing import Any

from jaxtyping import Float
from nanomanifold import SO3

from body_models._common import deformation, kinematics, skinning
from body_models._rotations import RotationType, rotation_ndim

Array = Any
Front = tuple[list[int], list[int]]


FlameSkeletonIdentity = deformation.SkeletonIdentity
FlameIdentity = deformation.LinearIdentity
FlamePreparedPose = deformation.SkinningPose


def prepare_pose(
    posedirs: Float[Array, "P V*3"],
    kinematic_fronts: list[Front],
    head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
    head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*batch J 3"],
    rest_joints: Float[Array, "*batch J 3"],
    xp: Any,
) -> FlamePreparedPose:
    """Prepare FLAME transforms and pose-dependent vertex offsets."""
    num_rot_dims = rotation_ndim(rotation_type)
    batch_shape = tuple(head_pose.shape[: -(num_rot_dims + 1)])
    local_joint_offsets = xp.broadcast_to(local_joint_offsets, (*batch_shape, *local_joint_offsets.shape[-2:]))
    rest_joints = xp.broadcast_to(rest_joints, (*batch_shape, *rest_joints.shape[-2:]))

    pose_matrices, world_transforms = _forward_core(
        xp=xp,
        kinematic_fronts=kinematic_fronts,
        local_joint_offsets=local_joint_offsets,
        head_pose=head_pose,
        head_rotation=head_rotation,
        rotation_type=rotation_type,
    )
    return {
        "skeleton_transforms": world_transforms,
        "skinning_transforms": skinning.bind_relative_transforms(world_transforms, rest_joints, xp=xp),
        "pose_offsets": deformation.pose_blend_shapes(pose_matrices, posedirs, xp=xp),
    }


def prepare_skeleton(
    kinematic_fronts: list[Front],
    head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
    head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*batch J 3"],
    xp: Any,
) -> Float[Array, "*batch 5 4 4"]:
    """Prepare only posed FLAME joint transforms."""
    batch_shape = head_pose.shape[: -(rotation_ndim(rotation_type) + 1)]
    local_joint_offsets = xp.broadcast_to(local_joint_offsets, (*batch_shape, *local_joint_offsets.shape[-2:]))
    _, transforms = _forward_core(
        xp=xp,
        kinematic_fronts=kinematic_fronts,
        local_joint_offsets=local_joint_offsets,
        head_pose=head_pose,
        head_rotation=head_rotation,
        rotation_type=rotation_type,
    )
    return transforms


def _forward_core(
    xp: Any,
    kinematic_fronts: list[Front],
    local_joint_offsets: Float[Array, "*batch J 3"],
    head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
    head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
) -> tuple[Float[Array, "*batch 5 3 3"], Float[Array, "*batch 5 4 4"]]:
    """Compose FLAME local rotations and run forward kinematics."""
    num_rot_dims = rotation_ndim(rotation_type)
    batch_shape = tuple(head_pose.shape[: -(num_rot_dims + 1)])
    head_matrices = SO3.convert(head_pose, src=rotation_type, dst="rotmat", xp=xp)
    if head_rotation is None:
        root_matrices = SO3.identity_as(
            head_matrices,
            batch_dims=(*batch_shape, 1),
            rotation_type="rotmat",
            xp=xp,
        )
    else:
        root_matrices = SO3.convert(
            head_rotation,
            src=rotation_type,
            dst="rotmat",
            xp=xp,
        )[..., None, :, :]
    pose_matrices = xp.concat([root_matrices, head_matrices], axis=-3)
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
    j_template: Float[Array, "5 3"],
    j_shapedirs: Float[Array, "5 3 S"],
    j_exprdirs: Float[Array, "5 3 E"],
    parents: list[int],
    shape: Float[Array, "*batch S"],
    expression: Float[Array, "*batch E"],
) -> FlameIdentity:
    """Prepare shape- and expression-dependent FLAME state."""
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
    j_template: Float[Array, "5 3"],
    j_shapedirs: Float[Array, "5 3 S"],
    j_exprdirs: Float[Array, "5 3 E"],
    parents: list[int],
    shape: Float[Array, "*batch S"],
    expression: Float[Array, "*batch E"],
) -> FlameSkeletonIdentity:
    """Prepare only identity-dependent FLAME joint state."""
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


__all__ = ["FlameIdentity", "FlamePreparedPose", "prepare_identity", "prepare_pose"]
