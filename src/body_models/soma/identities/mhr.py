"""MHR identity setup for SOMA."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaxtyping import Float

from ... import common
from ..io import SomaIdentityTransfer, get_identity_model_path


@dataclass(frozen=True)
class MHRIdentity:
    model_path: Path


def prepare(transfer: SomaIdentityTransfer) -> tuple[MHRIdentity, SomaIdentityTransfer]:
    model_path = get_identity_model_path("mhr")
    if model_path is None:
        raise ValueError("SOMA model_type='mhr' requires a configured MHR model path.")
    return MHRIdentity(model_path=model_path), transfer


def prepare_backend_model(identity_model: MHRIdentity, backend: str) -> Any:
    if backend == "numpy":
        from ...mhr.numpy import MHR
    elif backend == "torch":
        from ...mhr.torch import MHR
    elif backend == "jax":
        from flax import nnx

        from ...mhr.jax import MHR

        return nnx.data(MHR(model_path=identity_model.model_path, simplify=1.0))
    else:
        raise ValueError(f"Unsupported MHR identity backend target: {backend}")

    return MHR(model_path=identity_model.model_path, simplify=1.0)


def shape(
    identity_model: Any,
    identity: Float[Any, "B I"],
    scale_params: Float[Any, "B K"] | None,
    num_scale_params: int,
    *,
    xp: Any,
) -> Float[Any, "B V 3"]:
    batch_size = identity.shape[0]
    if scale_params is None:
        scale_params = common.zeros_as(identity, shape=(batch_size, num_scale_params), xp=xp)
    zero_pose = common.zeros_as(identity, shape=(batch_size, identity_model.pose_dim), xp=xp)
    zero_pose = common.set(zero_pose, (slice(None), slice(-num_scale_params, None)), scale_params, xp=xp)
    expression = common.zeros_as(identity, shape=(batch_size, identity_model.EXPR_DIM), xp=xp)
    return identity_model.forward_vertices(shape=identity, pose=zero_pose, expression=expression)
