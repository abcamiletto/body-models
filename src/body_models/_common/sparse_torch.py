"""PyTorch lowering of sparse linear weights."""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor, nn

from body_models._common import sparse as sparse_common


class SparseLinear(nn.Module):
    """Sparse linear weights backed by the native sparse matrix multiply."""

    transpose: Tensor

    def __init__(self, weights: sparse_common.SparseMatrix) -> None:
        super().__init__()
        row_indices = torch.as_tensor(weights.row_indices)
        column_indices = torch.as_tensor(weights.column_indices)
        values = torch.as_tensor(weights.values)
        indices = torch.stack((column_indices, row_indices))
        transpose = torch.sparse_coo_tensor(
            indices,
            values,
            (weights.shape[1], weights.shape[0]),
        ).coalesce()
        self.register_buffer("transpose", transpose, persistent=True)

    def forward(
        self,
        inputs: Float[Tensor, "*batch input"],
    ) -> Float[Tensor, "*batch output"]:
        batch_shape = inputs.shape[:-1]
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        output = _sparse_mm(self.transpose, flat_inputs.T).T
        return output.reshape(*batch_shape, self.transpose.shape[0])


@torch.compiler.disable
def _sparse_mm(
    matrix: Float[Tensor, "output input"],
    inputs: Float[Tensor, "input B"],
) -> Float[Tensor, "output B"]:
    """Keep the native sparse operator as one explicit compiled graph boundary."""
    return torch.sparse.mm(matrix, inputs)


__all__ = ["SparseLinear"]
