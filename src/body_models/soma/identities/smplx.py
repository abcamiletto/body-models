"""SMPL-X identity setup for SOMA."""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from jaxtyping import Float

from ...smplx.numpy import SMPLX
from ..io import SomaIdentityTransfer, get_identity_model_path


@dataclass(frozen=True)
class SMPLXIdentity:
    mean: Float[np.ndarray, "V 3"]
    shapedirs: Float[np.ndarray, "V 3 S"]


def prepare(transfer: SomaIdentityTransfer) -> tuple[SMPLXIdentity, SomaIdentityTransfer]:
    model = SMPLX(model_path=get_identity_model_path("smplx"), simplify=1.0)
    identity_model = SMPLXIdentity(mean=model.v_template_full, shapedirs=model.shapedirs_full)
    return identity_model, transfer


def prepare_backend_model(identity_model: SMPLXIdentity, backend: str) -> SMPLXIdentity:
    if backend == "numpy":
        return identity_model
    if backend == "torch":
        import torch

        return replace(
            identity_model,
            mean=torch.as_tensor(identity_model.mean),
            shapedirs=torch.as_tensor(identity_model.shapedirs),
        )
    if backend == "jax":
        import jax.numpy as jnp

        return replace(
            identity_model,
            mean=jnp.asarray(identity_model.mean),
            shapedirs=jnp.asarray(identity_model.shapedirs),
        )
    raise ValueError(f"Unsupported SMPL-X identity backend target: {backend}")


def shape(
    identity_model: SMPLXIdentity,
    identity: Float[Any, "B I"],
    scale_params: Float[Any, "B K"] | None = None,
    num_scale_params: int | None = None,
    *,
    xp: Any,
) -> Float[Any, "B V 3"]:
    identity_dim = identity.shape[1]
    return identity_model.mean[None] + xp.einsum("bi,vci->bvc", identity, identity_model.shapedirs[..., :identity_dim])
