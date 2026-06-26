"""MJCF-backed SMPL humanoid robot support."""

from body_models.robots.smpl_humanoid.constants import SMPL_HUMANOID_VARIANTS
from body_models.robots.smpl_humanoid.io import (
    SMPL_HUMANOID_SOURCES,
    SmplHumanoidWeights,
    load_model_data,
)

__all__ = [
    "SMPL_HUMANOID_SOURCES",
    "SMPL_HUMANOID_VARIANTS",
    "SmplHumanoidWeights",
    "load_model_data",
]
