"""Pose packing helpers for MHR."""

from typing import Any

from jaxtyping import Float

from body_models import _pose_layout as pose_layout

Array = Any

POSE_LAYOUT = pose_layout.PoseLayout(
    (
        ("body_pose", 27),
        ("head_pose", 3),
        ("body_pose", 38),
        ("hand_pose", 54),
        ("body_pose", 14),
        ("head_pose", 3),
        ("body_pose", 15),
        ("hand_pose", 50),
    )
)


def _require_last_dim(name: str, value: Float[Array, "... N"], size: int) -> None:
    if value.ndim < 1 or value.shape[-1] != size:
        raise ValueError(f"{name} must have shape [..., {size}], got {tuple(value.shape)}")


def pack_pose(
    xp: Any,
    body_pose: Float[Array, "... 94"],
    head_pose: Float[Array, "... 6"],
    hand_pose: Float[Array, "... 104"],
) -> Float[Array, "... 204"]:
    """Pack separated MHR body/head/hand controls into the canonical 204-vector."""
    _require_last_dim("body_pose", body_pose, 94)
    _require_last_dim("head_pose", head_pose, 6)
    _require_last_dim("hand_pose", hand_pose, 104)
    values = {"body_pose": body_pose, "head_pose": head_pose, "hand_pose": hand_pose}
    return POSE_LAYOUT.pack(xp, values, axis=-1)


def unpack_pose(
    xp: Any,
    pose: Float[Array, "... 204"],
) -> tuple[Float[Array, "... 94"], Float[Array, "... 6"], Float[Array, "... 104"]]:
    """Split the canonical MHR 204-vector into body, head, and hand controls."""
    _require_last_dim("pose", pose, 204)
    unpacked = POSE_LAYOUT.unpack(xp, pose, axis=-1)
    return unpacked["body_pose"], unpacked["head_pose"], unpacked["hand_pose"]


__all__ = ["POSE_LAYOUT", "pack_pose", "unpack_pose"]
