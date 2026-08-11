"""PyTorch SmplHumanoid model."""

from body_models._backend import model_for_backend
from body_models.smpl_humanoid._model import SmplHumanoid as _SmplHumanoid

SmplHumanoid = model_for_backend(_SmplHumanoid, "torch", module=__name__)

__all__ = ["SmplHumanoid"]
