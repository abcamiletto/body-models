"""Public SMPL-X API."""

from body_models.smplx._core import JointRegressor
from body_models.smplx._model import SMPLX

SMPLX.__module__ = __name__
JointRegressor.__module__ = __name__

__all__ = ["SMPLX", "JointRegressor"]
