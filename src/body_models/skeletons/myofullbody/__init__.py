"""MyoFullBody musculoskeletal model support.

The model is sourced from ``amathislab/musclemimic_models`` (Apache-2.0). It is
exposed through the same multi-backend ``forward_skeleton``/``forward_meshes``
API as other rigid-body models, with STL link meshes attached to a
MuJoCo-derived body tree. Inputs are scalar joint coordinates (``body_pose``)
plus a free-root ``global_translation``/``global_rotation``.
"""

from typing import Any

from .backends import core as _core

__all__ = ["to_qpos", "from_mujoco_qpos"]


def to_qpos(
    model: Any,
    body_pose: Any,
    global_translation: Any | None = None,
    *,
    global_rotation: Any | None = None,
) -> Any:
    """Convert MyoFullBody body-models inputs to MuJoCo ``qpos`` (length ``7 + Q``)."""
    return model.to_qpos(
        body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
    )


def from_mujoco_qpos(qpos: Any) -> dict[str, Any]:
    """Split a MuJoCo ``qpos`` into ``body_pose``/``global_rotation``/``global_translation``."""
    return _core.from_mujoco_qpos(qpos)
