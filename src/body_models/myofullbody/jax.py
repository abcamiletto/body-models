"""JAX MyoFullBody model."""

from body_models._backend import model_for_backend
from body_models.myofullbody._model import MyoFullBody as _MyoFullBody

MyoFullBody = model_for_backend(_MyoFullBody, "jax", module=__name__)

__all__ = ["MyoFullBody"]
