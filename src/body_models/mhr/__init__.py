"""Public MHR API."""

from body_models.mhr._model import MHR, MhrIdentity, MhrPreparedPose

MHR.__module__ = __name__

__all__ = ["MHR", "MhrIdentity", "MhrPreparedPose"]
