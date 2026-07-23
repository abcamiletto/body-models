"""SMPL humanoid Euler-joint kinematics."""

from __future__ import annotations

from typing import Any

from jaxtyping import Float
from nanomanifold import SO3
from body_models.common import rigid

Array = Any


def _body_rotations(
    body_pose: Float[Array, "B Q"],
    num_actuated_joints: int,
    *,
    xp: Any,
) -> Float[Array, "B A 3 3"]:
    euler = body_pose.reshape(*body_pose.shape[:-1], num_actuated_joints, 3)
    return SO3.conversions.from_euler_to_rotmat(euler, convention="XYZ", xp=xp)


def forward_skeleton(
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    actuated_joint_indices: list[int],
    parents: list[int],
    body_pose: Float[Array, "B Q"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    xp: Any,
) -> Float[Array, "B J 4 4"]:
    """Compute world-space SMPL humanoid joint transforms."""
    body_rot = _body_rotations(body_pose, len(actuated_joint_indices), xp=xp)
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


__all__ = ["forward_skeleton"]
