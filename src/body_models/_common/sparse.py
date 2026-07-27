"""Backend-neutral sparse linear weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from jaxtyping import Float, Int

Array = Any


@dataclass(frozen=True)
class SparseMatrix:
    """A two-dimensional matrix in coordinate format."""

    row_indices: Int[np.ndarray, "NNZ"]
    column_indices: Int[np.ndarray, "NNZ"]
    values: Float[np.ndarray, "NNZ"]
    shape: tuple[int, int]


class SparseLinear(Protocol):
    """Backend materialization of sparse weights shaped ``[input, output]``."""

    def __call__(
        self,
        inputs: Float[Array, "*batch input"],
    ) -> Float[Array, "*batch output"]: ...


def from_dense(matrix: Float[np.ndarray, "input output"]) -> SparseMatrix:
    """Store the exact nonzero entries of a dense matrix."""
    rows, columns = np.nonzero(matrix)
    return SparseMatrix(
        row_indices=rows.astype(np.int64, copy=False),
        column_indices=columns.astype(np.int64, copy=False),
        values=matrix[rows, columns],
        shape=matrix.shape,
    )


def linear(
    inputs: Float[Array, "*batch input"],
    weights: SparseLinear,
) -> Float[Array, "*batch output"]:
    """Apply sparse weights to the final dimension of ``inputs``."""
    return weights(inputs)


__all__ = ["SparseLinear", "SparseMatrix", "from_dense", "linear"]
