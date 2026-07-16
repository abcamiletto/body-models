"""Contract and shared feature extraction for SOMA's corrective network."""

from __future__ import annotations

from typing import Any, Protocol

from jaxtyping import Float

from body_models.common import set

Array = Any


class CorrectiveNetwork(Protocol):
    """Evaluate SOMA's learned pose-corrective network."""

    def __call__(
        self,
        hidden: Float[Array, "*batch H"],
    ) -> Float[Array, "*batch Vf 3"]: ...


def hidden_activations(
    pose_rotations: Float[Array, "*batch J 3 3"],
    bindpose: Float[Array, "J 3 3"],
    weights: Float[Array, "J*6 H"],
    *,
    xp: Any,
) -> Float[Array, "*batch H"]:
    """Evaluate the dense first layer shared by all sparse output lowerings."""
    batch_shape = pose_rotations.shape[:-3]
    relative = bindpose.swapaxes(-2, -1) @ pose_rotations
    features = relative[..., :, :, :2]
    features = set(features, (..., slice(None), 0, 0), features[..., :, 0, 0] - 1, xp=xp)
    features = set(features, (..., slice(None), 1, 1), features[..., :, 1, 1] - 1, xp=xp)
    features = features.reshape(*batch_shape, -1)
    hidden = features @ weights
    return xp.maximum(hidden, xp.zeros_like(hidden))


__all__ = ["CorrectiveNetwork", "hidden_activations"]
