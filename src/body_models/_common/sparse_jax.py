"""JAX lowering of sparse linear weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jaxtyping import Float, Int

from body_models._common import sparse as sparse_common

Array = Any


@dataclass(frozen=True)
class SparseLinear:
    """Sparse linear weights evaluated by indexed accumulation."""

    row_indices: Int[Array, "NNZ"]
    column_indices: Int[Array, "NNZ"]
    values: Float[Array, "NNZ"]
    output_size: int

    @classmethod
    def from_matrix(cls, weights: sparse_common.SparseMatrix) -> SparseLinear:
        return cls(
            row_indices=jnp.asarray(weights.row_indices),
            column_indices=jnp.asarray(weights.column_indices),
            values=jnp.asarray(weights.values),
            output_size=weights.shape[1],
        )

    def __call__(
        self,
        inputs: Float[Array, "*batch input"],
    ) -> Float[Array, "*batch output"]:
        contributions = inputs[..., self.row_indices] * self.values
        output = jnp.zeros_like(inputs, shape=(*inputs.shape[:-1], self.output_size))
        return output.at[..., self.column_indices].add(contributions)


__all__ = ["SparseLinear"]
