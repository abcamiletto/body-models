"""Public SMPL API."""

from body_models.smpl._model import SMPL, SmplIdentity, SmplPreparedPose

SMPL.__module__ = __name__

__all__ = ["SMPL", "SmplIdentity", "SmplPreparedPose"]
