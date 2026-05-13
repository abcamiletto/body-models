"""Pose packing helpers for GarmentMeasurements."""

from typing import Any

from jaxtyping import Float

Array = Any


def _joint_axis(pose: Array) -> int:
    return -3 if pose.shape[-2:] == (3, 3) else -2


def pack_pose(
    xp: Any,
    pelvis_rotation: Float[Array, "... N"] | Float[Array, "... 3 3"],
    body_pose: Float[Array, "... 25 N"] | Float[Array, "... 25 3 3"],
    head_pose: Float[Array, "... 3 N"] | Float[Array, "... 3 3 3"],
    hand_pose: Float[Array, "... 30 N"] | Float[Array, "... 30 3 3"],
) -> Float[Array, "... 59 N"] | Float[Array, "... 59 3 3"]:
    """Pack separated GarmentMeasurements pose groups into the canonical 59-joint pose."""
    joint_axis = _joint_axis(body_pose)
    if joint_axis == -3:
        pelvis = pelvis_rotation[..., None, :, :]
        spine = body_pose[..., :5, :, :]
        left_arm = body_pose[..., 5:11, :, :]
        right_arm = body_pose[..., 11:17, :, :]
        legs = body_pose[..., 17:, :, :]
        left_hand = hand_pose[..., :15, :, :]
        right_hand = hand_pose[..., 15:, :, :]
    else:
        pelvis = pelvis_rotation[..., None, :]
        spine = body_pose[..., :5, :]
        left_arm = body_pose[..., 5:11, :]
        right_arm = body_pose[..., 11:17, :]
        legs = body_pose[..., 17:, :]
        left_hand = hand_pose[..., :15, :]
        right_hand = hand_pose[..., 15:, :]

    return xp.concat([pelvis, spine, head_pose, left_arm, left_hand, right_arm, right_hand, legs], axis=joint_axis)


def unpack_pose(
    xp: Any,
    pose: Float[Array, "... 59 N"] | Float[Array, "... 59 3 3"],
) -> tuple[
    Float[Array, "... N"] | Float[Array, "... 3 3"],
    Float[Array, "... 25 N"] | Float[Array, "... 25 3 3"],
    Float[Array, "... 3 N"] | Float[Array, "... 3 3 3"],
    Float[Array, "... 30 N"] | Float[Array, "... 30 3 3"],
]:
    """Split the canonical GarmentMeasurements pose into pelvis, body, head, and hands."""
    joint_axis = _joint_axis(pose)
    if joint_axis == -3:
        pelvis_rotation = pose[..., 0, :, :]
        body_pose = xp.concat(
            [pose[..., 1:6, :, :], pose[..., 9:15, :, :], pose[..., 30:36, :, :], pose[..., 51:59, :, :]],
            axis=joint_axis,
        )
        head_pose = pose[..., 6:9, :, :]
        hand_pose = xp.concat([pose[..., 15:30, :, :], pose[..., 36:51, :, :]], axis=joint_axis)
    else:
        pelvis_rotation = pose[..., 0, :]
        body_pose = xp.concat(
            [pose[..., 1:6, :], pose[..., 9:15, :], pose[..., 30:36, :], pose[..., 51:59, :]],
            axis=joint_axis,
        )
        head_pose = pose[..., 6:9, :]
        hand_pose = xp.concat([pose[..., 15:30, :], pose[..., 36:51, :]], axis=joint_axis)
    return pelvis_rotation, body_pose, head_pose, hand_pose


__all__ = ["pack_pose", "unpack_pose"]
