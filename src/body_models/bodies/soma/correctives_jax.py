"""JAX scatter lowering of SOMA's corrective network."""

from __future__ import annotations

from typing import Any

from body_models.bodies.soma.correctives import hidden_activations
from body_models.runtime import Runtime


class JaxCorrectiveNetwork:
    """SOMA corrective network using a JAX indexed reduction."""

    def __init__(self, runtime: Runtime, data: Any) -> None:
        del data
        self._runtime = runtime

    def __call__(self, data: Any, pose_rotations: Any) -> Any:
        hidden = hidden_activations(self._runtime, data, pose_rotations)
        correctives = data.correctives
        contributions = hidden[..., correctives.corrective_W2_rows] * correctives.corrective_W2_values
        output_size = data.mean_full.shape[0] * 3
        output = self._runtime.zeros((*hidden.shape[:-1], output_size), like=hidden)
        output = output.at[..., correctives.corrective_W2_cols].add(contributions)
        return output.reshape(*hidden.shape[:-1], data.mean_full.shape[0], 3)


__all__ = ["JaxCorrectiveNetwork"]
