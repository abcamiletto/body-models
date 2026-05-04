"""MHR identity setup for SOMA."""

from .base import MhrIdentityData
from ..io import SomaIdentityTransfer, get_identity_model_path


def prepare(transfer: SomaIdentityTransfer) -> tuple[MhrIdentityData, SomaIdentityTransfer]:
    return MhrIdentityData(model_path=get_identity_model_path("mhr")), transfer
