"""Public SMPL-X API."""

from body_models.smplx._model import SMPLX, SmplxIdentity, SmplxPreparedPose

SMPLX.__module__ = __name__

__all__ = ["SMPLX", "SmplxIdentity", "SmplxPreparedPose"]
