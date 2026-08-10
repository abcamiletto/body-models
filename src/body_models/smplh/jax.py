"""JAX SMPL-H model."""

from body_models._backend import model_for_backend
from body_models.smplh._model import SMPLH as _SMPLH

SMPLH = model_for_backend(_SMPLH, "jax", module=__name__)

__all__ = ["SMPLH"]
