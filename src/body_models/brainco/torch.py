"""PyTorch BrainCo model."""

from body_models._backend import model_for_backend
from body_models.brainco._model import BrainCoHand as _BrainCoHand

BrainCoHand = model_for_backend(_BrainCoHand, "torch", module=__name__)

__all__ = ["BrainCoHand"]
