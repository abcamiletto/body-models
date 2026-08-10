"""NumPy MANO model."""

from body_models._backend import model_for_backend
from body_models.mano._model import MANO as _MANO

MANO = model_for_backend(_MANO, "numpy", module=__name__)

__all__ = ["MANO"]
