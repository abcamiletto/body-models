"""SciPy sparse lowering of SOMA's corrective network."""

from __future__ import annotations

from typing import Any

from jaxtyping import Float
from scipy import sparse

from body_models.runtime import ArrayRuntime

Array = Any


class NumpyCorrectiveNetwork:
    """SOMA corrective network backed by a SciPy CSR output layer."""

    def __init__(self, runtime: ArrayRuntime, data: Any) -> None:
        correctives = data.correctives
        shape = correctives.corrective_W1.shape[1], data.mean_full.shape[0] * 3
        indices = correctives.corrective_W2_rows, correctives.corrective_W2_cols
        self._matrix = sparse.csr_matrix((correctives.corrective_W2_values, indices), shape=shape)
        self._num_vertices = data.mean_full.shape[0]
        self._runtime = runtime

    def __call__(
        self,
        hidden: Float[Array, "*batch H"],
    ) -> Float[Array, "*batch Vf 3"]:
        batch_shape = hidden.shape[:-1]
        output = hidden.reshape(-1, hidden.shape[-1]) @ self._matrix
        return self._runtime.xp.asarray(output).reshape(*batch_shape, self._num_vertices, 3)


__all__ = ["NumpyCorrectiveNetwork"]
