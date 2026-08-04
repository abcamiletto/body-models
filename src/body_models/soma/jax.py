"""JAX SOMA model."""

from body_models._backend import model_for_backend
from body_models.soma import SOMA as _SOMA

SOMA = model_for_backend(_SOMA, "jax", module=__name__)

__all__ = ["SOMA"]
