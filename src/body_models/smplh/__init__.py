"""Public SMPL-H API."""

from body_models.smplh._model import SMPLH, SmplhIdentity, SmplhPreparedPose

SMPLH.__module__ = __name__

__all__ = ["SMPLH", "SmplhIdentity", "SmplhPreparedPose"]
