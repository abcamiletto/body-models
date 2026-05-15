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
    rotation_dims = (slice(None), slice(None)) if joint_axis == -3 else (slice(None),)
    pose_parts = [
        pelvis_rotation[(..., None, *rotation_dims)],
        body_pose[(..., slice(None, 5), *rotation_dims)],
        head_pose,
        body_pose[(..., slice(5, 11), *rotation_dims)],
        hand_pose[(..., slice(None, 15), *rotation_dims)],
        body_pose[(..., slice(11, 17), *rotation_dims)],
        hand_pose[(..., slice(15, None), *rotation_dims)],
        body_pose[(..., slice(17, None), *rotation_dims)],
    ]
    return xp.concat(pose_parts, axis=joint_axis)


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
    rotation_dims = (slice(None), slice(None)) if joint_axis == -3 else (slice(None),)
    pelvis_rotation = pose[(..., 0, *rotation_dims)]
    body_parts = [
        pose[(..., slice(1, 6), *rotation_dims)],
        pose[(..., slice(9, 15), *rotation_dims)],
        pose[(..., slice(30, 36), *rotation_dims)],
        pose[(..., slice(51, 59), *rotation_dims)],
    ]
    hand_parts = [
        pose[(..., slice(15, 30), *rotation_dims)],
        pose[(..., slice(36, 51), *rotation_dims)],
    ]
    body_pose = xp.concat(body_parts, axis=joint_axis)
    head_pose = pose[(..., slice(6, 9), *rotation_dims)]
    hand_pose = xp.concat(hand_parts, axis=joint_axis)
    return pelvis_rotation, body_pose, head_pose, hand_pose


__all__ = ["pack_pose", "unpack_pose"]
