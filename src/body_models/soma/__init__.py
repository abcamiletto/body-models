"""Public SOMA API."""

from body_models.soma._model import SOMA, SomaIdentity, SomaPreparedPose

SOMA.__module__ = __name__

__all__ = ["SOMA", "SomaIdentity", "SomaPreparedPose"]
