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

    row_indices: Int[Array, "NNZ"]
    column_indices: Int[Array, "NNZ"]
    values: Float[Array, "NNZ"]
    shape: tuple[int, int]


class SparseLinear(Protocol):
    """Backend materialization of sparse weights shaped ``[input, output]``."""

    def __call__(
        self,
        inputs: Float[Array, "*batch input"],
    ) -> Float[Array, "*batch output"]: ...

    @property
    def shape(self) -> tuple[int, int]: ...

    def to_coo(self) -> SparseMatrix: ...


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


def select_columns(
    matrix: SparseMatrix,
    columns: Int[np.ndarray, "output"],
) -> SparseMatrix:
    """Select and reorder output columns of a sparse linear map."""
    column_map = np.full(matrix.shape[1], -1, dtype=np.int64)
    column_map[columns] = np.arange(columns.size)
    remapped = column_map[matrix.column_indices]
    keep = remapped >= 0
    return SparseMatrix(
        row_indices=matrix.row_indices[keep],
        column_indices=remapped[keep],
        values=matrix.values[keep],
        shape=(matrix.shape[0], columns.size),
    )


def scaled(matrix: SparseMatrix, factor: float) -> SparseMatrix:
    """Scale the outputs of a sparse linear map."""
    return SparseMatrix(
        row_indices=matrix.row_indices,
        column_indices=matrix.column_indices,
        values=matrix.values * factor,
        shape=matrix.shape,
    )


__all__ = ["SparseLinear", "SparseMatrix", "from_dense", "linear", "scaled", "select_columns"]
