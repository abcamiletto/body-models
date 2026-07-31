"""Public MANO API."""

from body_models.mano._model import MANO, ManoIdentity, ManoPreparedPose

MANO.__module__ = __name__

__all__ = ["MANO", "ManoIdentity", "ManoPreparedPose"]
