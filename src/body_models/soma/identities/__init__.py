"""SOMA identity setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jaxtyping import Float

from . import mhr, smpl, smplx
from ..io import SomaIdentityTransfer

IDENTITY_BACKENDS = {
    "mhr": mhr,
    "smpl": smpl,
    "smplx": smplx,
}

__all__ = ["IDENTITY_BACKENDS", "IdentityBackend", "load", "shape"]


@dataclass(frozen=True)
class IdentityBackend:
    model_type: str
    identity_dim: int
    num_scale_params: int | None
    default_identity_value: float
    model: Any = None
    transfer: SomaIdentityTransfer | None = None


def load(model_type: str, spec: Any, transfer: SomaIdentityTransfer | None = None) -> IdentityBackend:
    if model_type == "soma":
        return IdentityBackend(
            model_type=model_type,
            identity_dim=spec.identity_dim,
            num_scale_params=spec.num_scale_params,
            default_identity_value=spec.default_identity_value,
        )

    backend = IDENTITY_BACKENDS[model_type]
    model, transfer = backend.prepare(transfer)
    return IdentityBackend(
        model_type=model_type,
        identity_dim=spec.identity_dim,
        num_scale_params=spec.num_scale_params,
        default_identity_value=spec.default_identity_value,
        model=model,
        transfer=transfer,
    )


def shape(
    *,
    backend: IdentityBackend,
    identity: Float[Any, "B I"],
    scale_params: Float[Any, "B K"] | None,
    xp: Any,
) -> Float[Any, "B V 3"]:
    model_type = backend.model_type
    module = IDENTITY_BACKENDS[model_type]

    if model_type == "mhr":
        return module.shape(
            identity_model=backend.model,
            identity=identity,
            scale_params=scale_params,
            num_scale_params=backend.num_scale_params,
            xp=xp,
        )
    return module.shape(identity_model=backend.model, identity=identity, xp=xp)
