"""PyTorch lowering of sparse linear weights."""

from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from body_models.common import sparse as sparse_common


class SparseLinear(nn.Module):
    """Sparse linear weights with eager and compile-safe multiplication."""

    transpose: Tensor
    row_indices: Int[Tensor, "NNZ"]
    column_indices: Int[Tensor, "NNZ"]
    values: Float[Tensor, "NNZ"]

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
        self.register_buffer("transpose", transpose)
        self.register_buffer("row_indices", row_indices)
        self.register_buffer("column_indices", column_indices)
        self.register_buffer("values", values)
        self._output_size = weights.shape[1]

    def forward(
        self,
        inputs: Float[Tensor, "*batch input"],
    ) -> Float[Tensor, "*batch output"]:
        batch_shape = inputs.shape[:-1]
        if torch.compiler.is_compiling():
            contributions = inputs[..., self.row_indices] * self.values
            output = torch.zeros(
                (*batch_shape, self._output_size),
                dtype=inputs.dtype,
                device=inputs.device,
            )
            output_indices = torch.broadcast_to(self.column_indices, contributions.shape)
            return output.scatter_add(-1, output_indices, contributions)

        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        output = torch.sparse.mm(self.transpose, flat_inputs.T).T
        return output.reshape(*batch_shape, self._output_size)


__all__ = ["SparseLinear"]
