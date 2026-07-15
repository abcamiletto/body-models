"""Unitree G1 hinge kinematics."""

from __future__ import annotations

from typing import Any, Literal

from jaxtyping import Float
from nanomanifold import SO3

from body_models.common import rigid

Array = Any
Convention = Literal["soma", "mujoco"]

MUJOCO_TO_KIMODO = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))


def _hinge_rotations(
    body_pose: Float[Array, "B Q"],
    actuated_joint_axes: Float[Array, "Q 3"],
    *,
    xp: Any,
) -> Float[Array, "B Q 3 3"]:
    if body_pose.ndim < 1 or body_pose.shape[-1] != actuated_joint_axes.shape[0]:
        raise ValueError(
            f"G1 body_pose must have shape [..., {actuated_joint_axes.shape[0]}], got {tuple(body_pose.shape)}"
        )
    axes = xp.asarray(actuated_joint_axes, dtype=body_pose.dtype)
    return SO3.convert(body_pose[..., None], src="hinge", dst="rotmat", src_kwargs={"axes": axes}, xp=xp)


def forward_skeleton(
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    actuated_joint_indices: list[int],
    actuated_joint_axes: Float[Array, "Q 3"],
    parents: list[int],
    body_pose: Float[Array, "B Q"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    xp: Any,
) -> Float[Array, "B J 4 4"]:
    """Compute world-space G1 joint transforms from local rotations."""
    body_rot = _hinge_rotations(body_pose, actuated_joint_axes, xp=xp)
    return rigid.forward_skeleton_from_local_rotations(
        body_rot,
        local_offsets=local_offsets,
        rest_local_rotations=rest_local_rotations,
        actuated_joint_indices=actuated_joint_indices,
        parents=parents,
        global_translation=global_translation,
        global_rotation=global_rotation,
        joint_indices=joint_indices,
        xp=xp,
    )
