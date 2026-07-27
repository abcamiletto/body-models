"""Public SmplHumanoid API."""

from body_models.smpl_humanoid._constants import SMPL_HUMANOID_VARIANTS
from body_models.smpl_humanoid._model import SmplHumanoid

SmplHumanoid.__module__ = __name__

__all__ = [
    "SMPL_HUMANOID_VARIANTS",
    "SmplHumanoid",
]
