"""NumPy FLAME model."""

from body_models._backend import model_for_backend
from body_models.flame import FLAME as _FLAME

FLAME = model_for_backend(_FLAME, "numpy", module=__name__)

__all__ = ["FLAME"]
