"""NumPy MHR model."""

from body_models._backend import model_for_backend
from body_models.mhr._model import MHR as _MHR

MHR = model_for_backend(_MHR, "numpy", module=__name__)

__all__ = ["MHR"]
