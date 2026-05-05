"""SOMA identity setup."""

from __future__ import annotations

from typing import Any

from jaxtyping import Float

from . import mhr, smpl, smplx
from ..io import SomaIdentityTransfer

IDENTITY_BACKENDS = {
    "mhr": mhr,
    "smpl": smpl,
    "smplx": smplx,
}

__all__ = ["IDENTITY_BACKENDS", "IdentityBackend", "load", "replace_data", "shape"]


class IdentityBackend:
    def __init__(
        self,
        *,
        model_type: str,
        identity_dim: int,
        num_scale_params: int | None,
        default_identity_value: float,
        model: Any = None,
        transfer: SomaIdentityTransfer | None = None,
    ) -> None:
        self.model_type = model_type
        self.identity_dim = identity_dim
        self.num_scale_params = num_scale_params
        self.default_identity_value = default_identity_value
        self.model = model
        self.transfer = transfer


def load(model_type: str, spec: Any, transfer: SomaIdentityTransfer | None = None) -> IdentityBackend:
    if transfer is None:
        return IdentityBackend(
            model_type=model_type,
            identity_dim=spec.identity_dim,
            num_scale_params=spec.num_scale_params,
            default_identity_value=spec.default_identity_value,
        )

    try:
        backend = IDENTITY_BACKENDS[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported SOMA identity backend: {model_type}") from exc
    model, transfer = backend.prepare(transfer)
    return IdentityBackend(
        model_type=model_type,
        identity_dim=spec.identity_dim,
        num_scale_params=spec.num_scale_params,
        default_identity_value=spec.default_identity_value,
        model=model,
        transfer=transfer,
    )


def replace_data(
    backend: IdentityBackend,
    *,
    model: Any | None = None,
    transfer: SomaIdentityTransfer | None = None,
) -> IdentityBackend:
    return IdentityBackend(
        model_type=backend.model_type,
        identity_dim=backend.identity_dim,
        num_scale_params=backend.num_scale_params,
        default_identity_value=backend.default_identity_value,
        model=backend.model if model is None else model,
        transfer=backend.transfer if transfer is None else transfer,
    )


def shape(
    *,
    backend: IdentityBackend,
    identity: Float[Any, "B I"],
    scale_params: Float[Any, "B K"] | None,
    xp: Any,
) -> Float[Any, "B V 3"]:
    model_type = backend.model_type
    try:
        module = IDENTITY_BACKENDS[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported SOMA identity backend: {model_type}") from exc

    if model_type == "mhr":
        if backend.num_scale_params is None:
            raise ValueError("SOMA model_type='mhr' requires num_scale_params.")
        return module.shape(
            identity_model=backend.model,
            identity=identity,
            scale_params=scale_params,
            num_scale_params=backend.num_scale_params,
            xp=xp,
        )
    return module.shape(identity_model=backend.model, identity=identity, xp=xp)
