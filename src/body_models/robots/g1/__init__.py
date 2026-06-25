"""Unitree G1 rigid articulated model support."""

from typing import Any

__all__ = ["to_mujoco_qpos"]


def to_mujoco_qpos(
    model: Any,
    body_pose: Any,
    global_translation: Any | None = None,
    *,
    global_rotation: Any | None = None,
    clamp_to_limits: bool = False,
) -> Any:
    """Convert a G1 body pose to MuJoCo qpos."""
    return model.to_mujoco_qpos(
        body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        clamp_to_limits=clamp_to_limits,
    )
