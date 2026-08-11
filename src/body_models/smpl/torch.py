"""PyTorch SMPL model."""

from body_models._backend import model_for_backend
from body_models.smpl._model import SMPL as _SMPL

SMPL = model_for_backend(_SMPL, "torch", module=__name__)

__all__ = ["SMPL"]
