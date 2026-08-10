"""JAX SKEL model."""

from body_models._backend import model_for_backend
from body_models.skel._model import SKEL as _SKEL

SKEL = model_for_backend(_SKEL, "jax", module=__name__)

__all__ = ["SKEL"]
