"""JAX GNM Head model."""

from body_models._backend import model_for_backend
from body_models.gnm._model import GNM as _GNM

GNM = model_for_backend(_GNM, "jax", module=__name__)

__all__ = ["GNM"]
