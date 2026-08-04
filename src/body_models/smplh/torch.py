"""PyTorch SMPL-H model."""

from body_models._backend import model_for_backend
from body_models.smplh import SMPLH as _SMPLH

SMPLH = model_for_backend(_SMPLH, "torch", module=__name__)

__all__ = ["SMPLH"]
