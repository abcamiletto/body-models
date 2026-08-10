"""NumPy SMPL-X model."""

from body_models._backend import model_for_backend
from body_models.smplx._model import SMPLX as _SMPLX

SMPLX = model_for_backend(_SMPLX, "numpy", module=__name__)

__all__ = ["SMPLX"]
