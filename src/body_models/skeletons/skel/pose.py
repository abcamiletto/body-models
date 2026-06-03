"""Pose packing helpers for SKEL."""

from typing import Any

from jaxtyping import Float

Array = Any

SKEL_BODY_POSE_DIM = 43
SKEL_HEAD_POSE_DIM = 3
SKEL_BODY_HEAD_SPLIT = 23
SKEL_CANONICAL_POSE_DIM = 46


def pack_pose(
    xp: Any,
    body_pose: Float[Array, "... 43"],
    head_pose: Float[Array, "... 3"],
) -> Float[Array, "... 46"]:
    """Pack separated SKEL body/head controls into the canonical 46-vector."""
    return xp.concat(
        [
            body_pose[..., :SKEL_BODY_HEAD_SPLIT],
            head_pose,
            body_pose[..., SKEL_BODY_HEAD_SPLIT:],
        ],
        axis=-1,
    )


def unpack_pose(
    xp: Any,
    pose: Float[Array, "... 46"],
) -> tuple[Float[Array, "... 43"], Float[Array, "... 3"]]:
    """Split the canonical SKEL pose into body and head controls."""
    body_pose = xp.concat(
        [
            pose[..., :SKEL_BODY_HEAD_SPLIT],
            pose[..., SKEL_BODY_HEAD_SPLIT + SKEL_HEAD_POSE_DIM :],
        ],
        axis=-1,
    )
    head_pose = pose[..., SKEL_BODY_HEAD_SPLIT : SKEL_BODY_HEAD_SPLIT + SKEL_HEAD_POSE_DIM]
    return body_pose, head_pose


__all__ = [
    "SKEL_BODY_POSE_DIM",
    "SKEL_HEAD_POSE_DIM",
    "SKEL_CANONICAL_POSE_DIM",
    "pack_pose",
    "unpack_pose",
]
