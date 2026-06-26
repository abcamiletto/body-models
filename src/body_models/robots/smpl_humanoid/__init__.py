"""MJCF-backed SMPL humanoid robot support."""

from body_models.robots.smpl_humanoid.io import (
    SMPL_HUMANOID_MODEL_TYPES,
    SmplHumanoidWeights,
    load_model_data,
)

__all__ = [
    "SMPL_HUMANOID_MODEL_TYPES",
    "SmplHumanoidWeights",
    "load_model_data",
]
