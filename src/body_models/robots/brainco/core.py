"""BrainCo Revo 2 coupled-joint kinematics."""

from __future__ import annotations

from typing import Any

from jaxtyping import Float
from nanomanifold import SO3

from body_models import common
from body_models.common import rigid

Array = Any
MUJOCO_TO_KIMODO = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))


def _hinge_rotations(
    pose: Float[Array, "B Q"],
    actuated_joint_axes: Float[Array, "Q 3"],
    *,
    xp: Any,
) -> Float[Array, "B Q 3 3"]:
    if pose.ndim < 1 or pose.shape[-1] != actuated_joint_axes.shape[0]:
        raise ValueError(f"BrainCo pose must have shape [..., {actuated_joint_axes.shape[0]}], got {tuple(pose.shape)}")
    axes = xp.asarray(actuated_joint_axes, dtype=pose.dtype)
    return SO3.convert(pose[..., None], src="hinge", dst="rotmat", src_kwargs={"axes": axes}, xp=xp)


def forward_skeleton(
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    actuated_joint_axes: Float[Array, "Q 3"],
    actuated_joint_indices: list[int],
    coupled_joint_axes: Float[Array, "C 3"],
    coupled_joint_indices: list[int],
    coupled_driver_indices: list[int],
    coupled_polycoef: Float[Array, "C 4"],
    parents: list[int],
    pose: Float[Array, "B Q"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    xp: Any,
) -> Float[Array, "B J 4 4"]:
    """Compute world-space BrainCo hand joint transforms."""
    local_joint_rot = _hinge_rotations(pose, actuated_joint_axes, xp=xp)

    batch_shape = tuple(local_joint_rot.shape[:-3])
    dtype = local_joint_rot.dtype
    num_joints = len(parents)
    if global_translation is None:
        global_translation = common.zeros_as(local_joint_rot, shape=(*batch_shape, 3), xp=xp)

    rest_rot = xp.asarray(rest_local_rotations, dtype=dtype)
    local_rot = common.eye_as(local_joint_rot, batch_dims=(*batch_shape, num_joints), xp=xp)
    local_rot = common.set(local_rot, (..., actuated_joint_indices, slice(None), slice(None)), local_joint_rot, xp=xp)
    if coupled_joint_indices:
        driver_pose = pose[..., coupled_driver_indices]
        coeffs = xp.asarray(coupled_polycoef, dtype=dtype)
        coupled_pose = (
            coeffs[:, 0]
            + coeffs[:, 1] * driver_pose
            + coeffs[:, 2] * driver_pose * driver_pose
            + coeffs[:, 3] * driver_pose * driver_pose * driver_pose
        )
        coupled_rot = SO3.convert(
            coupled_pose[..., None],
            src="hinge",
            dst="rotmat",
            src_kwargs={"axes": xp.asarray(coupled_joint_axes, dtype=dtype)},
            xp=xp,
        )
        local_rot = common.set(local_rot, (..., coupled_joint_indices, slice(None), slice(None)), coupled_rot, xp=xp)
    local_rot = xp.broadcast_to(rest_rot, (*batch_shape, num_joints, 3, 3)) @ local_rot
    return rigid.forward_skeleton_from_local_transforms(
        local_rot,
        local_offsets=local_offsets,
        parents=parents,
        global_translation=global_translation,
        global_rotation=global_rotation,
        joint_indices=joint_indices,
        xp=xp,
    )
