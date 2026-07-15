"""Contract and shared feature extraction for SOMA's corrective network."""

from __future__ import annotations

from typing import Any, Protocol

from body_models.runtime import Runtime


class CorrectiveNetwork(Protocol):
    """Evaluate SOMA's learned pose-corrective network."""

    def __call__(self, data: Any, pose_rotations: Any) -> Any: ...


def create_corrective_network(runtime: Runtime, data: Any) -> CorrectiveNetwork:
    """Build the efficient corrective-network lowering for a runtime."""
    if runtime.name == "numpy":
        from body_models.bodies.soma.correctives_numpy import NumpyCorrectiveNetwork

        return NumpyCorrectiveNetwork(data)
    if runtime.name == "torch":
        from body_models.bodies.soma.correctives_torch import TorchCorrectiveNetwork

        return TorchCorrectiveNetwork(data)
    if runtime.name == "jax":
        from body_models.bodies.soma.correctives_jax import JaxCorrectiveNetwork

        return JaxCorrectiveNetwork()
    raise ValueError(f"SOMA does not support runtime {runtime.name!r}")


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


__all__ = ["CorrectiveNetwork", "create_corrective_network", "hidden_activations"]
