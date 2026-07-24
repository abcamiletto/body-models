"""Public SmplHumanoid API."""

from body_models.robots.smpl_humanoid import (
    SMPL_HUMANOID_SOURCES,
    SMPL_HUMANOID_VARIANTS,
    SmplHumanoidWeights,
    download_model,
    get_model_path,
    load_model_data,
    validate_path,
)
from body_models.rigid import RigidBodyParameters as Parameters

__all__ = [
    "SMPL_HUMANOID_SOURCES",
    "SMPL_HUMANOID_VARIANTS",
    "Parameters",
    "SmplHumanoidWeights",
    "download_model",
    "get_model_path",
    "load_model_data",
    "validate_path",
]
