"""Pose packing helpers for SOMA."""

from typing import Any

from jaxtyping import Float

Array = Any


def _joint_axis(pose: Array) -> int:
    return -3 if pose.shape[-2:] == (3, 3) else -2


def pack_pose(
    xp: Any,
    pelvis_rotation: Float[Array, "... N"] | Float[Array, "... 3 3"],
    body_pose: Float[Array, "... 23 N"] | Float[Array, "... 23 3 3"],
    head_pose: Float[Array, "... 5 N"] | Float[Array, "... 5 3 3"],
    hand_pose: Float[Array, "... 48 N"] | Float[Array, "... 48 3 3"],
) -> Float[Array, "... 77 N"] | Float[Array, "... 77 3 3"]:
    """Pack separated SOMA pose groups into the canonical 77-joint pose."""
    joint_axis = _joint_axis(body_pose)
    if joint_axis == -3:
        pelvis = pelvis_rotation[..., None, :, :]
        body_spine = body_pose[..., :5, :, :]
        body_left_arm = body_pose[..., 5:9, :, :]
        body_right_arm = body_pose[..., 9:13, :, :]
        body_legs = body_pose[..., 13:, :, :]
        left_hand = hand_pose[..., :24, :, :]
        right_hand = hand_pose[..., 24:, :, :]
    else:
        pelvis = pelvis_rotation[..., None, :]
        body_spine = body_pose[..., :5, :]
        body_left_arm = body_pose[..., 5:9, :]
        body_right_arm = body_pose[..., 9:13, :]
        body_legs = body_pose[..., 13:, :]
        left_hand = hand_pose[..., :24, :]
        right_hand = hand_pose[..., 24:, :]

    return xp.concat(
        [
            pelvis,
            body_spine,
            head_pose,
            body_left_arm,
            left_hand,
            body_right_arm,
            right_hand,
            body_legs,
        ],
        axis=joint_axis,
    )


def unpack_pose(
    xp: Any,
    pose: Float[Array, "... 77 N"] | Float[Array, "... 77 3 3"],
) -> tuple[
    Float[Array, "... N"] | Float[Array, "... 3 3"],
    Float[Array, "... 23 N"] | Float[Array, "... 23 3 3"],
    Float[Array, "... 5 N"] | Float[Array, "... 5 3 3"],
    Float[Array, "... 48 N"] | Float[Array, "... 48 3 3"],
]:
    """Split the canonical SOMA pose into pelvis, body, head, and hands."""
    joint_axis = _joint_axis(pose)
    if joint_axis == -3:
        pelvis_rotation = pose[..., 0, :, :]
        body_pose = xp.concat(
            [pose[..., 1:6, :, :], pose[..., 11:15, :, :], pose[..., 39:43, :, :], pose[..., 67:77, :, :]],
            axis=joint_axis,
        )
        head_pose = pose[..., 6:11, :, :]
        hand_pose = xp.concat([pose[..., 15:39, :, :], pose[..., 43:67, :, :]], axis=joint_axis)
    else:
        pelvis_rotation = pose[..., 0, :]
        body_pose = xp.concat(
            [pose[..., 1:6, :], pose[..., 11:15, :], pose[..., 39:43, :], pose[..., 67:77, :]],
            axis=joint_axis,
        )
        head_pose = pose[..., 6:11, :]
        hand_pose = xp.concat([pose[..., 15:39, :], pose[..., 43:67, :]], axis=joint_axis)
    return pelvis_rotation, body_pose, head_pose, hand_pose


__all__ = ["pack_pose", "unpack_pose"]
