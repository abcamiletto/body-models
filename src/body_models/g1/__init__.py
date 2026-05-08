"""Unitree G1 rigid articulated model support."""

from typing import Any

from body_models.g1.backends import core as _core

__all__ = ["to_mujoco_qpos"]


def to_mujoco_qpos(
    model: Any,
    body_pose: Any,
    global_translation: Any | None = None,
    *,
    global_rotation: Any | None = None,
    clamp_to_limits: bool = True,
) -> Any:
    """Convert a G1 body pose to MuJoCo qpos."""
    return _core.to_mujoco_qpos(
        qpos_joint_axes=model.qpos_joint_axes,
        qpos_joint_limits=model.qpos_joint_limits,
        body_pose=body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        clamp_to_limits=clamp_to_limits,
        rotation_type=model.rotation_type,
        convention=model.convention,
    )
