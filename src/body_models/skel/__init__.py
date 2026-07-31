"""Public SKEL API."""

from body_models.skel._model import SKEL, SkelIdentity, SkelPreparedPose

SKEL.__module__ = __name__

__all__ = ["SKEL", "SkelIdentity", "SkelPreparedPose"]
