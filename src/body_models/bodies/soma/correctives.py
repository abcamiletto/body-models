"""Contract and shared feature extraction for SOMA's corrective network."""

from __future__ import annotations

from typing import Any, Protocol

from body_models.runtime import Runtime


class CorrectiveNetwork(Protocol):
    """Evaluate SOMA's learned pose-corrective network."""

    def __call__(self, data: Any, pose_rotations: Any) -> Any: ...


def hidden_activations(runtime: Runtime, data: Any, pose_rotations: Any) -> Any:
    """Evaluate the dense first layer shared by all sparse output lowerings."""
    correctives = data.correctives
    batch_shape = pose_rotations.shape[:-3]
    relative = correctives.corrective_bindpose.swapaxes(-2, -1) @ pose_rotations
    identity_columns = runtime.asarray(
        [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
        like=relative,
    )
    features = (relative[..., :, :, :2] - identity_columns).reshape(*batch_shape, -1)
    hidden = features @ correctives.corrective_W1
    return runtime.xp.maximum(hidden, runtime.asarray(0.0, like=hidden))


__all__ = ["CorrectiveNetwork", "hidden_activations"]
