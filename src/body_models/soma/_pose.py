"""Pose packing helpers for SOMA."""

from typing import Any

from jaxtyping import Float

from body_models import _pose_layout as pose_layout

Array = Any

POSE_LAYOUT = pose_layout.PoseLayout.per_joint(
    ("root_rotation", 1),
    ("body_pose", 5),
    ("head_pose", 5),
    ("body_pose", 4),
    ("hand_pose", 24),
    ("body_pose", 4),
    ("hand_pose", 24),
    ("body_pose", 10),
)


def _joint_axis(pose: Float[Array, "..."]) -> int:
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
    pose: Float[Array, "... 77 N"] | Float[Array, "... 77 3 3"],
) -> tuple[
    Float[Array, "... N"] | Float[Array, "... 3 3"],
    Float[Array, "... 23 N"] | Float[Array, "... 23 3 3"],
    Float[Array, "... 5 N"] | Float[Array, "... 5 3 3"],
    Float[Array, "... 48 N"] | Float[Array, "... 48 3 3"],
]:
    """Split the canonical SOMA pose into global rotation, body, head, and hands."""
    joint_axis = _joint_axis(pose)
    unpacked = POSE_LAYOUT.unpack(xp, pose, axis=joint_axis)
    global_rotation = xp.squeeze(unpacked["root_rotation"], axis=joint_axis)
    return global_rotation, unpacked["body_pose"], unpacked["head_pose"], unpacked["hand_pose"]


__all__ = ["POSE_LAYOUT", "pack_pose", "unpack_pose"]
