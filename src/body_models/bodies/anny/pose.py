"""Pose packing helpers for ANNY."""

from typing import Any

from jaxtyping import Float

Array = Any
DEFAULT_NUM_JOINTS = 163
DEFAULT_POSE_GROUP_SIZES = (64, 60, 38)


def _joint_axis(pose: Array) -> int:
    return -3 if pose.shape[-2:] == (3, 3) else -2


def pack_pose(
    xp: Any,
    global_rotation: Float[Array, "... N"] | Float[Array, "... 3 3"],
    body_pose: Float[Array, "... 64 N"] | Float[Array, "... 64 3 3"],
    head_pose: Float[Array, "... 60 N"] | Float[Array, "... 60 3 3"],
    hand_pose: Float[Array, "... 38 N"] | Float[Array, "... 38 3 3"],
) -> Float[Array, "... J N"] | Float[Array, "... J 3 3"]:
    """Pack separated ANNY pose groups into this rig's native pose."""
    joint_axis = _joint_axis(body_pose)
    rotation_dims = (slice(None), slice(None)) if joint_axis == -3 else (slice(None),)
    root = global_rotation[(..., None, *rotation_dims)]

    group_sizes = (body_pose.shape[joint_axis], head_pose.shape[joint_axis], hand_pose.shape[joint_axis])
    if group_sizes != DEFAULT_POSE_GROUP_SIZES:
        return xp.concat([root, body_pose, head_pose, hand_pose], axis=joint_axis)

    return xp.concat(
        [
            root,
            body_pose[(..., slice(None, 54), *rotation_dims)],
            hand_pose[(..., slice(None, 19), *rotation_dims)],
            body_pose[(..., slice(54, 61), *rotation_dims)],
            hand_pose[(..., slice(19, None), *rotation_dims)],
            body_pose[(..., slice(61, None), *rotation_dims)],
            head_pose,
        ],
        axis=joint_axis,
    )


def unpack_pose(
    xp: Any,
    pose: Float[Array, "... 163 N"] | Float[Array, "... 163 3 3"],
) -> tuple[
    Float[Array, "... N"] | Float[Array, "... 3 3"],
    Float[Array, "... 64 N"] | Float[Array, "... 64 3 3"],
    Float[Array, "... 60 N"] | Float[Array, "... 60 3 3"],
    Float[Array, "... 38 N"] | Float[Array, "... 38 3 3"],
]:
    """Split an ANNY pose into global rotation, body, head, and hands."""
    joint_axis = _joint_axis(pose)
    rotation_dims = (slice(None), slice(None)) if joint_axis == -3 else (slice(None),)
    global_rotation = pose[(..., 0, *rotation_dims)]

    if pose.shape[joint_axis] != DEFAULT_NUM_JOINTS:
        body_pose = pose[(..., slice(1, None), *rotation_dims)]
        head_pose = pose[(..., slice(1, 1), *rotation_dims)]
        hand_pose = pose[(..., slice(1, 1), *rotation_dims)]
        return global_rotation, body_pose, head_pose, hand_pose

    body_parts = [
        pose[(..., slice(1, 55), *rotation_dims)],
        pose[(..., slice(74, 81), *rotation_dims)],
        pose[(..., slice(100, 103), *rotation_dims)],
    ]
    hand_parts = [
        pose[(..., slice(55, 74), *rotation_dims)],
        pose[(..., slice(81, 100), *rotation_dims)],
    ]
    body_pose = xp.concat(body_parts, axis=joint_axis)
    head_pose = pose[(..., slice(103, 163), *rotation_dims)]
    hand_pose = xp.concat(hand_parts, axis=joint_axis)
    return global_rotation, body_pose, head_pose, hand_pose


__all__ = ["pack_pose", "unpack_pose"]
