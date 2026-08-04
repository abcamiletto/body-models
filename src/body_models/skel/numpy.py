"""NumPy SKEL model."""

from body_models._backend import model_for_backend
from body_models.skel import SKEL as _SKEL

SKEL = model_for_backend(_SKEL, "numpy", module=__name__)

__all__ = ["SKEL"]
