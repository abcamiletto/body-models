"""Pose packing helpers for MHR."""

from typing import Any

from jaxtyping import Float

from body_models.bodies.mhr.constants import (
    MHR_BODY_EYE_SPLIT,
    MHR_BODY_HEAD_SPLIT,
    MHR_CANONICAL_BODY_POSE_DIM,
    MHR_BODY_POSE_SPLIT,
    MHR_HAND_POSE_SPLIT,
)

Array = Any


def pack_pose(
    xp: Any,
    body_pose: Float[Array, "... 94"],
    head_pose: Float[Array, "... 6"],
    hand_pose: Float[Array, "... 104"],
) -> Float[Array, "... 204"]:
    """Pack separated MHR body/head/hand controls into the canonical 204-vector."""
    body_pose = xp.concat(
        [
            body_pose[..., :MHR_BODY_HEAD_SPLIT],
            head_pose[..., :3],
            body_pose[..., MHR_BODY_HEAD_SPLIT:MHR_BODY_EYE_SPLIT],
            head_pose[..., 3:],
            body_pose[..., MHR_BODY_EYE_SPLIT:],
        ],
        axis=-1,
    )
    return xp.concat(
        [
            body_pose[..., :MHR_BODY_POSE_SPLIT],
            hand_pose[..., :MHR_HAND_POSE_SPLIT],
            body_pose[..., MHR_BODY_POSE_SPLIT:],
            hand_pose[..., MHR_HAND_POSE_SPLIT:],
        ],
        axis=-1,
    )


def unpack_pose(
    xp: Any,
    pose: Float[Array, "... 204"],
) -> tuple[Float[Array, "... 94"], Float[Array, "... 6"], Float[Array, "... 104"]]:
    """Split the canonical MHR 204-vector into body, head, and hand controls."""
    body_tail_start = MHR_BODY_POSE_SPLIT + MHR_HAND_POSE_SPLIT
    body_tail_end = body_tail_start + (MHR_CANONICAL_BODY_POSE_DIM - MHR_BODY_POSE_SPLIT)
    old_body_pose = xp.concat(
        [
            pose[..., :MHR_BODY_POSE_SPLIT],
            pose[..., body_tail_start:body_tail_end],
        ],
        axis=-1,
    )
    hand_pose = xp.concat(
        [
            pose[..., MHR_BODY_POSE_SPLIT : MHR_BODY_POSE_SPLIT + MHR_HAND_POSE_SPLIT],
            pose[..., body_tail_end:],
        ],
        axis=-1,
    )
    body_pose = xp.concat(
        [
            old_body_pose[..., :MHR_BODY_HEAD_SPLIT],
            old_body_pose[..., MHR_BODY_HEAD_SPLIT + 3 : MHR_BODY_EYE_SPLIT + 3],
            old_body_pose[..., MHR_BODY_EYE_SPLIT + 6 :],
        ],
        axis=-1,
    )
    head_pose = xp.concat(
        [
            old_body_pose[..., MHR_BODY_HEAD_SPLIT : MHR_BODY_HEAD_SPLIT + 3],
            old_body_pose[..., MHR_BODY_EYE_SPLIT + 3 : MHR_BODY_EYE_SPLIT + 6],
        ],
        axis=-1,
    )
    return body_pose, head_pose, hand_pose


__all__ = ["pack_pose", "unpack_pose"]
