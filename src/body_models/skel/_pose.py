"""Pose packing helpers for SKEL."""

from typing import Any

from jaxtyping import Float

from body_models import _pose_layout as pose_layout

Array = Any

SKEL_BODY_POSE_COEFFS = 43
SKEL_HEAD_POSE_COEFFS = 3
_BODY_HEAD_SPLIT = 23
JOINT_DOFS = (3, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1, 1, 2, 3, 3, 1, 1, 2)
_CONTROL_JOINTS = tuple((joint,) for joint, dofs in enumerate(JOINT_DOFS) for _ in range(dofs))
POSE_LAYOUT = pose_layout.PoseLayout(
    (
        ("body_pose", _BODY_HEAD_SPLIT),
        ("head_pose", SKEL_HEAD_POSE_COEFFS),
        ("body_pose", SKEL_BODY_POSE_COEFFS - _BODY_HEAD_SPLIT),
    ),
    _CONTROL_JOINTS,
)


def pack_pose(
    xp: Any,
    body_pose: Float[Array, "... 43"],
    head_pose: Float[Array, "... 3"],
) -> Float[Array, "... 46"]:
    """Pack separated SKEL body/head controls into the canonical 46-vector."""
    return POSE_LAYOUT.pack(xp, {"body_pose": body_pose, "head_pose": head_pose}, axis=-1)


def unpack_pose(
    xp: Any,
    pose: Float[Array, "... 46"],
) -> tuple[Float[Array, "... 43"], Float[Array, "... 3"]]:
    """Split the canonical SKEL pose into body and head controls."""
    unpacked = POSE_LAYOUT.unpack(xp, pose, axis=-1)
    return unpacked["body_pose"], unpacked["head_pose"]


__all__ = [
    "JOINT_DOFS",
    "POSE_LAYOUT",
    "SKEL_BODY_POSE_COEFFS",
    "SKEL_HEAD_POSE_COEFFS",
    "pack_pose",
    "unpack_pose",
]
