"""MyoFullBody musculoskeletal model support."""

from typing import Any

from array_api_compat import get_namespace
from jaxtyping import Float

from body_models.myofullbody import core

Array = Any


def from_mujoco_qpos(qpos: Float[Array, "*batch qpos"]) -> dict[str, Float[Array, "..."]]:
    """Split MuJoCo ``qpos`` into model pose parameters."""
    return core.from_mujoco_qpos(qpos, xp=get_namespace(qpos))


__all__ = ["from_mujoco_qpos"]
