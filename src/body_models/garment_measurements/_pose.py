"""Pose packing helpers for GarmentMeasurements."""

from typing import Any

from jaxtyping import Float

from body_models import _pose_layout as pose_layout

Array = Any

POSE_LAYOUT = pose_layout.PoseLayout.per_joint(
    ("pelvis_rotation", 1),
    ("body_pose", 5),
    ("head_pose", 3),
    ("body_pose", 6),
    ("hand_pose", 15),
    ("body_pose", 6),
    ("hand_pose", 15),
    ("body_pose", 8),
)


def _control_axis(pose: Float[Array, "..."]) -> int:
    return -3 if pose.shape[-2:] == (3, 3) else -2


def pack_pose(
    xp: Any,
    pelvis_rotation: Float[Array, "... N"] | Float[Array, "... 3 3"],
    body_pose: Float[Array, "... 25 N"] | Float[Array, "... 25 3 3"],
    head_pose: Float[Array, "... 3 N"] | Float[Array, "... 3 3 3"],
    hand_pose: Float[Array, "... 30 N"] | Float[Array, "... 30 3 3"],
) -> Float[Array, "... 59 N"] | Float[Array, "... 59 3 3"]:
    """Pack separated GarmentMeasurements pose groups into the canonical 59-control pose."""
    control_axis = _control_axis(body_pose)
    rotation_dims = (slice(None), slice(None)) if control_axis == -3 else (slice(None),)
    root = pelvis_rotation[(..., None, *rotation_dims)]
    return POSE_LAYOUT.pack(
        xp,
        {
            "pelvis_rotation": root,
            "body_pose": body_pose,
            "head_pose": head_pose,
            "hand_pose": hand_pose,
        },
        axis=control_axis,
    )


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
    control_axis = _control_axis(pose)
    unpacked = POSE_LAYOUT.unpack(xp, pose, axis=control_axis)
    pelvis_rotation = xp.squeeze(unpacked["pelvis_rotation"], axis=control_axis)
    return pelvis_rotation, unpacked["body_pose"], unpacked["head_pose"], unpacked["hand_pose"]


__all__ = ["POSE_LAYOUT", "pack_pose", "unpack_pose"]
