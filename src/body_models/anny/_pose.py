"""Pose packing and rotation-conversion helpers for ANNY."""

from collections.abc import Mapping
from typing import Any

from jaxtyping import Float
from nanomanifold import SO3

from body_models import _pose_layout as pose_layout
from body_models import _rotations as rotations

Array = Any

POSE_LAYOUT = pose_layout.PoseLayout.per_joint(
    ("root_rotation", 1),
    ("body_pose", 54),
    ("hand_pose", 19),
    ("body_pose", 7),
    ("hand_pose", 19),
    ("body_pose", 3),
    ("head_pose", 60),
)


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


def _joint_axis(pose: Float[Array, "..."]) -> int:
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

    return POSE_LAYOUT.pack(
        xp,
        {
            "root_rotation": root,
            "body_pose": body_pose,
            "head_pose": head_pose,
            "hand_pose": hand_pose,
        },
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
    unpacked = POSE_LAYOUT.unpack(xp, pose, axis=joint_axis)
    global_rotation = xp.squeeze(unpacked["root_rotation"], axis=joint_axis)
    return global_rotation, unpacked["body_pose"], unpacked["head_pose"], unpacked["hand_pose"]


__all__ = ["POSE_LAYOUT", "convert_pose", "pack_pose", "unpack_pose"]
