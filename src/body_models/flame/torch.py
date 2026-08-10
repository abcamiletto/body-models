"""PyTorch FLAME model."""

from body_models._backend import model_for_backend
from body_models.flame._model import FLAME as _FLAME

FLAME = model_for_backend(_FLAME, "torch", module=__name__)

__all__ = ["FLAME"]
