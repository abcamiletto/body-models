"""Public SKEL API."""

from body_models.skel._model import SKEL, SkelIdentity, SkelPose

SKEL.__module__ = __name__

__all__ = ["SKEL", "SkelIdentity", "SkelPose"]
