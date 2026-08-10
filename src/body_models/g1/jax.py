"""JAX G1 model."""

from body_models._backend import model_for_backend
from body_models.g1._model import G1 as _G1

G1 = model_for_backend(_G1, "jax", module=__name__)

__all__ = ["G1"]
