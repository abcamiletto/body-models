"""SMPL identity setup for SOMA."""

from ...smpl.numpy import SMPL
from .base import LinearIdentityData
from ..io import SomaIdentityTransfer, get_identity_model_path


def prepare(transfer: SomaIdentityTransfer) -> tuple[LinearIdentityData, SomaIdentityTransfer]:
    model = SMPL(model_path=get_identity_model_path("smpl"), simplify=1.0)
    identity_model = LinearIdentityData(mean=model.v_template_full, shapedirs=model.shapedirs_full)
    return identity_model, transfer
