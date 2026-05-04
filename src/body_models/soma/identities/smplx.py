"""SMPL-X identity setup for SOMA."""

from ...smplx.numpy import SMPLX
from .base import LinearIdentityData
from ..io import SomaIdentityTransfer, get_identity_model_path


def prepare(transfer: SomaIdentityTransfer) -> tuple[LinearIdentityData, SomaIdentityTransfer]:
    model = SMPLX(model_path=get_identity_model_path("smplx"), simplify=1.0)
    identity_model = LinearIdentityData(mean=model.v_template_full, shapedirs=model.shapedirs_full)
    return identity_model, transfer
