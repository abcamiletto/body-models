"""PyTorch ANNY model."""

from body_models._backend import model_for_backend
from body_models.anny._model import ANNY as _ANNY

ANNY = model_for_backend(_ANNY, "torch", module=__name__)

__all__ = ["ANNY"]
