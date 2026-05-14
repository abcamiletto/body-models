"""Pose packing helpers for SOMA."""

from typing import Any

from jaxtyping import Float
from nanomanifold import SO3

from body_models import common

Array = Any


def _joint_axis(pose: Array) -> int:
    return -3 if pose.shape[-2:] == (3, 3) else -2


def pack_pose(
    xp: Any,
    global_rotation: Float[Array, "... N"] | Float[Array, "... 3 3"],
    body_pose: Float[Array, "... 23 N"] | Float[Array, "... 23 3 3"],
    head_pose: Float[Array, "... 5 N"] | Float[Array, "... 5 3 3"],
    hand_pose: Float[Array, "... 48 N"] | Float[Array, "... 48 3 3"],
) -> Float[Array, "... 77 N"] | Float[Array, "... 77 3 3"]:
    """Pack separated SOMA pose groups into the canonical 77-joint pose."""
    joint_axis = _joint_axis(body_pose)
    rotation_dims = (slice(None), slice(None)) if joint_axis == -3 else (slice(None),)
    root = global_rotation[(..., None, *rotation_dims)]

    return xp.concat(
        [
            root,
            body_pose[(..., slice(None, 5), *rotation_dims)],
            head_pose,
            body_pose[(..., slice(5, 9), *rotation_dims)],
            hand_pose[(..., slice(None, 24), *rotation_dims)],
            body_pose[(..., slice(9, 13), *rotation_dims)],
            hand_pose[(..., slice(24, None), *rotation_dims)],
            body_pose[(..., slice(13, None), *rotation_dims)],
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
    """Split the canonical SOMA pose into global rotation, body, head, and hands."""
    joint_axis = _joint_axis(pose)
    rotation_dims = (slice(None), slice(None)) if joint_axis == -3 else (slice(None),)
    global_rotation = pose[(..., 0, *rotation_dims)]
    body_parts = [
        pose[(..., slice(1, 6), *rotation_dims)],
        pose[(..., slice(11, 15), *rotation_dims)],
        pose[(..., slice(39, 43), *rotation_dims)],
        pose[(..., slice(67, 77), *rotation_dims)],
    ]
    hand_parts = [
        pose[(..., slice(15, 39), *rotation_dims)],
        pose[(..., slice(43, 67), *rotation_dims)],
    ]
    body_pose = xp.concat(body_parts, axis=joint_axis)
    head_pose = pose[(..., slice(6, 11), *rotation_dims)]
    hand_pose = xp.concat(hand_parts, axis=joint_axis)
    return global_rotation, body_pose, head_pose, hand_pose


def relaxed_hand_pose(
    xp: Any,
    hand_pose: Float[Array, "... J N"] | Float[Array, "... J 3 3"],
    rotation_type: str,
    curl: float = 0.55,
) -> Float[Array, "... J N"] | Float[Array, "... J 3 3"]:
    template = hand_pose[..., :, 0, :] if hand_pose.shape[-2:] == (3, 3) else hand_pose
    axis_angle = common.set(xp.zeros_like(template), (..., 0), xp.asarray(curl, dtype=template.dtype), xp=xp)
    return SO3.convert(axis_angle, src="axis_angle", dst=rotation_type, xp=xp)


__all__ = ["pack_pose", "relaxed_hand_pose", "unpack_pose"]
