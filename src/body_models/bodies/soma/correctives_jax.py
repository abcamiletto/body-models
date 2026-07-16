"""JAX scatter lowering of SOMA's corrective network."""

from __future__ import annotations

from typing import Any

from jaxtyping import Float

from body_models.runtime import ArrayRuntime

Array = Any


class JaxCorrectiveNetwork:
    """SOMA corrective network using a JAX indexed reduction."""

    def __init__(self, runtime: ArrayRuntime, data: Any) -> None:
        correctives = data.correctives
        self._runtime = runtime
        self._rows = correctives.corrective_W2_rows
        self._cols = correctives.corrective_W2_cols
        self._values = correctives.corrective_W2_values
        self._num_vertices = data.mean_full.shape[0]

    def __call__(
        self,
        hidden: Float[Array, "*batch H"],
    ) -> Float[Array, "*batch Vf 3"]:
        contributions = hidden[..., self._rows] * self._values
        output_size = self._num_vertices * 3
        output = self._runtime.zeros((*hidden.shape[:-1], output_size), like=hidden)
        output = output.at[..., self._cols].add(contributions)
        return output.reshape(*hidden.shape[:-1], self._num_vertices, 3)


__all__ = ["JaxCorrectiveNetwork"]
