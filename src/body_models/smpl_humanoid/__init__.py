"""Public SmplHumanoid API."""

from body_models.smpl_humanoid._constants import SMPL_HUMANOID_VARIANTS
from body_models.smpl_humanoid._model import SmplHumanoid, SmplMannequin, SmplxMannequin
from body_models.smpl_humanoid.generate import generate_smpl_robot

SmplHumanoid.__module__ = __name__
SmplMannequin.__module__ = __name__
SmplxMannequin.__module__ = __name__

__all__ = [
    "SMPL_HUMANOID_VARIANTS",
    "SmplHumanoid",
    "SmplMannequin",
    "SmplxMannequin",
    "generate_smpl_robot",
]
