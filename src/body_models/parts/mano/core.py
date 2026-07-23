"""Backend-independent MANO pose and identity preparation."""

from typing import Any, TypedDict

from jaxtyping import Float
from nanomanifold import SO3

from body_models.common import deformation, kinematics, skinning
from body_models.rotations import RotationType, rotation_ndim

Array = Any
Front = tuple[list[int], list[int]]


class ManoSkeletonIdentity(TypedDict):
    """Shape-dependent joint state needed to pose the MANO skeleton."""

    rest_joints: Float[Array, "*batch J 3"]
    local_joint_offsets: Float[Array, "*batch J 3"]


class ManoIdentity(ManoSkeletonIdentity):
    """Complete shape-dependent MANO mesh state."""

    rest_vertices: Float[Array, "*batch V 3"]


class ManoPreparedPose(TypedDict):
    """Complete pose-dependent MANO mesh state."""

    skeleton_transforms: Float[Array, "*batch J 4 4"]
    skinning_transforms: Float[Array, "*batch J 4 4"]
    pose_offsets: Float[Array, "*batch V 3"]


def prepare_pose(
    posedirs: Float[Array, "P V*3"],
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "45"],
    hand_pose: Float[Array, "*batch 15 N"] | Float[Array, "*batch 15 3 3"],
    wrist_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*batch J 3"],
    rest_joints: Float[Array, "*batch J 3"],
    xp: Any,
) -> ManoPreparedPose:
    """Prepare MANO transforms and pose-dependent vertex offsets."""
    num_rot_dims = rotation_ndim(rotation_type)
    batch_shape = tuple(hand_pose.shape[: -(num_rot_dims + 1)])
    local_joint_offsets = xp.broadcast_to(local_joint_offsets, (*batch_shape, *local_joint_offsets.shape[-2:]))
    rest_joints = xp.broadcast_to(rest_joints, (*batch_shape, *rest_joints.shape[-2:]))

    pose_matrices, world_transforms = _forward_core(
        xp=xp,
        kinematic_fronts=kinematic_fronts,
        hand_mean=hand_mean,
        local_joint_offsets=local_joint_offsets,
        hand_pose=hand_pose,
        wrist_rotation=wrist_rotation,
        rotation_type=rotation_type,
    )
    return {
        "skeleton_transforms": world_transforms,
        "skinning_transforms": skinning.bind_relative_transforms(world_transforms, rest_joints, xp=xp),
        "pose_offsets": deformation.pose_blend_shapes(pose_matrices, posedirs, xp=xp),
    }


def prepare_skeleton(
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "45"],
    hand_pose: Float[Array, "*batch 15 N"] | Float[Array, "*batch 15 3 3"],
    wrist_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*batch J 3"],
    xp: Any,
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed MANO joint transforms."""
    batch_shape = hand_pose.shape[: -(rotation_ndim(rotation_type) + 1)]
    local_joint_offsets = xp.broadcast_to(local_joint_offsets, (*batch_shape, *local_joint_offsets.shape[-2:]))
    _, transforms = _forward_core(
        xp=xp,
        kinematic_fronts=kinematic_fronts,
        hand_mean=hand_mean,
        local_joint_offsets=local_joint_offsets,
        hand_pose=hand_pose,
        wrist_rotation=wrist_rotation,
        rotation_type=rotation_type,
    )
    return transforms


def _forward_core(
    xp: Any,
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "45"],
    local_joint_offsets: Float[Array, "*batch J 3"],
    hand_pose: Float[Array, "*batch 15 N"] | Float[Array, "*batch 15 3 3"],
    wrist_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
) -> tuple[Float[Array, "*batch J 3 3"], Float[Array, "*batch J 4 4"]]:
    """Compose MANO local rotations and run forward kinematics."""
    num_rot_dims = rotation_ndim(rotation_type)
    batch_shape = tuple(hand_pose.shape[: -(num_rot_dims + 1)])

    if rotation_type == "axis_angle":
        hand_axis_angle = hand_pose
    else:
        hand_axis_angle = SO3.convert(hand_pose, src=rotation_type, dst="axis_angle", xp=xp)
    hand_axis_angle = hand_axis_angle + hand_mean.reshape(15, 3)
    hand_matrices = SO3.convert(hand_axis_angle, src="axis_angle", dst="rotmat", xp=xp)

    if wrist_rotation is None:
        wrist_matrices = SO3.identity_as(
            hand_matrices,
            batch_dims=(*batch_shape, 1),
            rotation_type="rotmat",
            xp=xp,
        )
    else:
        wrist_matrices = SO3.convert(
            wrist_rotation,
            src=rotation_type,
            dst="rotmat",
            xp=xp,
        )[..., None, :, :]
    pose_matrices = xp.concat([wrist_matrices, hand_matrices], axis=-3)
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
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: list[int],
    shape: Float[Array, "*batch S"],
) -> ManoIdentity:
    """Prepare shape-dependent MANO joints and vertices."""
    identity = prepare_skeleton_identity(
        xp=xp,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        parents=parents,
        shape=shape,
    )
    shape_directions = shapedirs[:, :, : shape.shape[-1]]
    return {
        "rest_joints": identity["rest_joints"],
        "local_joint_offsets": identity["local_joint_offsets"],
        "rest_vertices": deformation.blend_shapes(v_template, shape_directions, shape, xp=xp),
    }


def prepare_skeleton_identity(
    *,
    xp: Any,
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: list[int],
    shape: Float[Array, "*batch S"],
) -> ManoSkeletonIdentity:
    """Prepare only shape-dependent MANO joint state."""
    if shape.ndim < 1 or shape.shape[-1] < 1:
        raise ValueError("shape must have shape [..., S] with S >= 1")

    joint_directions = j_shapedirs[:, :, : shape.shape[-1]]
    rest_joints = deformation.blend_shapes(j_template, joint_directions, shape, xp=xp)
    return {
        "rest_joints": rest_joints,
        "local_joint_offsets": kinematics.local_joint_offsets(rest_joints, parents, xp=xp),
    }


__all__ = ["ManoIdentity", "ManoPreparedPose", "prepare_identity", "prepare_pose"]
