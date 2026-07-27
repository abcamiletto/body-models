"""Public SmplHumanoid API."""

from body_models.smpl_humanoid._constants import SMPL_HUMANOID_VARIANTS
from body_models.smpl_humanoid._io import (
    SMPL_HUMANOID_SOURCES,
    SmplHumanoidWeights,
    download_model,
    get_model_path,
    load_model_data,
    validate_path,
)
from body_models.smpl_humanoid._model import SmplHumanoid

SmplHumanoid.__module__ = __name__

__all__ = [
    "SMPL_HUMANOID_SOURCES",
    "SMPL_HUMANOID_VARIANTS",
    "SmplHumanoid",
    "SmplHumanoidWeights",
    "download_model",
    "get_model_path",
    "load_model_data",
    "validate_path",
]
