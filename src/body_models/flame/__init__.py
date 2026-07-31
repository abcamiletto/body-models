"""Public FLAME API."""

from body_models.flame._model import FLAME, FlameIdentity, FlamePreparedPose

FLAME.__module__ = __name__

__all__ = ["FLAME", "FlameIdentity", "FlamePreparedPose"]
