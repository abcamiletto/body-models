"""Public SOMA API."""

from body_models.soma._model import SOMA, SomaIdentity, SomaPose

SOMA.__module__ = __name__

__all__ = ["SOMA", "SomaIdentity", "SomaPose"]
