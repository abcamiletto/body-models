"""Pose packing helpers for MHR."""

from typing import Any

from jaxtyping import Float

Array = Any

_BODY_HEAD_SPLIT = 27
_BODY_EYE_SPLIT = 79
_BODY_POSE_SPLIT = 68
_HAND_POSE_SPLIT = 54
_CANONICAL_BODY_POSE_DIM = 100


def pack_pose(
    xp: Any,
    body_pose: Float[Array, "... 94"],
    head_pose: Float[Array, "... 6"],
    hand_pose: Float[Array, "... 104"],
) -> Float[Array, "... 204"]:
    """Pack separated MHR body/head/hand controls into the canonical 204-vector."""
    body_pose = xp.concat(
        [
            body_pose[..., :_BODY_HEAD_SPLIT],
            head_pose[..., :3],
            body_pose[..., _BODY_HEAD_SPLIT:_BODY_EYE_SPLIT],
            head_pose[..., 3:],
            body_pose[..., _BODY_EYE_SPLIT:],
        ],
        axis=-1,
    )
    return xp.concat(
        [
            body_pose[..., :_BODY_POSE_SPLIT],
            hand_pose[..., :_HAND_POSE_SPLIT],
            body_pose[..., _BODY_POSE_SPLIT:],
            hand_pose[..., _HAND_POSE_SPLIT:],
        ],
        axis=-1,
    )


def unpack_pose(
    xp: Any,
    pose: Float[Array, "... 204"],
) -> tuple[Float[Array, "... 94"], Float[Array, "... 6"], Float[Array, "... 104"]]:
    """Split the canonical MHR 204-vector into body, head, and hand controls."""
    body_tail_start = _BODY_POSE_SPLIT + _HAND_POSE_SPLIT
    body_tail_end = body_tail_start + (_CANONICAL_BODY_POSE_DIM - _BODY_POSE_SPLIT)
    old_body_pose = xp.concat(
        [
            pose[..., :_BODY_POSE_SPLIT],
            pose[..., body_tail_start:body_tail_end],
        ],
        axis=-1,
    )
    hand_pose = xp.concat(
        [
            pose[..., _BODY_POSE_SPLIT : _BODY_POSE_SPLIT + _HAND_POSE_SPLIT],
            pose[..., body_tail_end:],
        ],
        axis=-1,
    )
    body_pose = xp.concat(
        [
            old_body_pose[..., :_BODY_HEAD_SPLIT],
            old_body_pose[..., _BODY_HEAD_SPLIT + 3 : _BODY_EYE_SPLIT + 3],
            old_body_pose[..., _BODY_EYE_SPLIT + 6 :],
        ],
        axis=-1,
    )
    head_pose = xp.concat(
        [
            old_body_pose[..., _BODY_HEAD_SPLIT : _BODY_HEAD_SPLIT + 3],
            old_body_pose[..., _BODY_EYE_SPLIT + 3 : _BODY_EYE_SPLIT + 6],
        ],
        axis=-1,
    )
    return body_pose, head_pose, hand_pose


__all__ = ["pack_pose", "unpack_pose"]
