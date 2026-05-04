"""SOMA identity setup."""

from __future__ import annotations

from typing import Any

from jaxtyping import Float

from . import anny, mhr, smpl, smplx
from ..io import SomaIdentityTransfer

IDENTITY_BACKENDS = {
    "anny": anny,
    "mhr": mhr,
    "smpl": smpl,
    "smplx": smplx,
}

__all__ = ["IDENTITY_BACKENDS", "prepare_backend", "shape"]


def prepare_backend(model_type: str, transfer: SomaIdentityTransfer) -> tuple[Any, SomaIdentityTransfer]:
    try:
        backend = IDENTITY_BACKENDS[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported SOMA identity backend: {model_type}") from exc
    return backend.prepare(transfer)


def shape(
    *,
    model_type: str,
    identity_model: Any,
    identity: Float[Any, "B I"],
    scale_params: Float[Any, "B K"] | None,
    num_scale_params: int | None,
    xp: Any,
) -> Float[Any, "B V 3"]:
    try:
        backend = IDENTITY_BACKENDS[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported SOMA identity backend: {model_type}") from exc

    if model_type == "mhr":
        if num_scale_params is None:
            raise ValueError("SOMA model_type='mhr' requires num_scale_params.")
        return backend.shape(
            identity_model=identity_model,
            identity=identity,
            scale_params=scale_params,
            num_scale_params=num_scale_params,
            xp=xp,
        )
    return backend.shape(identity_model=identity_model, identity=identity, xp=xp)
