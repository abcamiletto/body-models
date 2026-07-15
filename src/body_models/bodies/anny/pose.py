"""Pose packing and rotation-conversion helpers for ANNY."""

from collections.abc import Mapping
from typing import Any

from jaxtyping import Float
from nanomanifold import SO3

from body_models import rotations

Array = Any


def convert_pose(
    parameters: Mapping[str, Any],
    *,
    src: rotations.RotationType,
    dst: rotations.RotationType,
) -> dict[str, Any]:
    """Convert the rotations in an ANNY parameter dictionary."""
    converted = dict(parameters)
    for key in ("body_pose", "head_pose", "hand_pose", "global_rotation"):
        value = converted.get(key)
        if value is not None:
            converted[key] = SO3.convert(value, src=src, dst=dst)
    return converted


def _joint_axis(pose: Array) -> int:
    return -3 if pose.shape[-2:] == (3, 3) else -2


def pack_pose(
    xp: Any,
    global_rotation: Float[Array, "... N"] | Float[Array, "... 3 3"],
    body_pose: Float[Array, "... 64 N"] | Float[Array, "... 64 3 3"],
    head_pose: Float[Array, "... 60 N"] | Float[Array, "... 60 3 3"],
    hand_pose: Float[Array, "... 38 N"] | Float[Array, "... 38 3 3"],
) -> Float[Array, "... 163 N"] | Float[Array, "... 163 3 3"]:
    """Pack separated ANNY pose groups into the canonical 163-joint pose."""
    joint_axis = _joint_axis(body_pose)
    rotation_dims = (slice(None), slice(None)) if joint_axis == -3 else (slice(None),)
    root = global_rotation[(..., None, *rotation_dims)]

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
    """Split the canonical ANNY pose into global rotation, body, head, and hands."""
    joint_axis = _joint_axis(pose)
    rotation_dims = (slice(None), slice(None)) if joint_axis == -3 else (slice(None),)
    global_rotation = pose[(..., 0, *rotation_dims)]
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


__all__ = ["convert_pose", "pack_pose", "unpack_pose"]
