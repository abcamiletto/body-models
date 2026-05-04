"""SMPL identity setup for SOMA."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from jaxtyping import Float

from ...smpl.numpy import SMPL
from ..io import SomaIdentityTransfer, get_identity_model_path


@dataclass(frozen=True)
class SMPLIdentity:
    mean: Float[np.ndarray, "V 3"]
    shapedirs: Float[np.ndarray, "V 3 S"]


def prepare(transfer: SomaIdentityTransfer) -> tuple[SMPLIdentity, SomaIdentityTransfer]:
    model = SMPL(model_path=get_identity_model_path("smpl"), simplify=1.0)
    identity_model = SMPLIdentity(mean=model.v_template_full, shapedirs=model.shapedirs_full)
    return identity_model, transfer


def shape(
    identity_model: SMPLIdentity,
    identity: Float[Any, "B I"],
    *,
    xp: Any,
) -> Float[Any, "B V 3"]:
    identity_dim = identity.shape[1]
    return identity_model.mean[None] + xp.einsum("bi,vci->bvc", identity, identity_model.shapedirs[..., :identity_dim])
