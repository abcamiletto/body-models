"""Pose packing helpers for MHR."""

from typing import Any

from jaxtyping import Float

from body_models.mhr.constants import MHR_BODY_POSE_DIM, MHR_BODY_POSE_SPLIT, MHR_HAND_POSE_SPLIT

Array = Any


def pack_pose(
    xp: Any,
    body_pose: Float[Array, "... 100"],
    hand_pose: Float[Array, "... 104"],
) -> Float[Array, "... 204"]:
    """Pack separated MHR body/hand controls into the canonical 204-vector."""
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
) -> tuple[Float[Array, "... 100"], Float[Array, "... 104"]]:
    """Split the canonical MHR 204-vector into body and hand controls."""
    body_tail_start = MHR_BODY_POSE_SPLIT + MHR_HAND_POSE_SPLIT
    body_tail_end = body_tail_start + (MHR_BODY_POSE_DIM - MHR_BODY_POSE_SPLIT)
    body_pose = xp.concat(
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
    return body_pose, hand_pose


__all__ = ["pack_pose", "unpack_pose"]
