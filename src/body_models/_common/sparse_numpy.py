"""NumPy lowering of sparse linear weights."""

from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Float
from scipy import sparse

from body_models._common import sparse as sparse_common

Array = Any


class SparseLinear:
    """Sparse linear weights backed by SciPy CSR."""

    def __init__(self, weights: sparse_common.SparseMatrix) -> None:
        indices = weights.row_indices, weights.column_indices
        self._weights = sparse.csr_array((weights.values, indices), shape=weights.shape)

    def __call__(
        self,
        inputs: Float[Array, "*batch input"],
    ) -> Float[Array, "*batch output"]:
        batch_shape = inputs.shape[:-1]
        output = inputs.reshape(-1, inputs.shape[-1]) @ self._weights
        return np.asarray(output).reshape(*batch_shape, self._weights.shape[1])


__all__ = ["SparseLinear"]
