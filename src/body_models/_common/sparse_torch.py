"""PyTorch lowering of sparse linear weights."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from jaxtyping import Float
from torch import Tensor, nn

from body_models._common import sparse as sparse_common

_LIBRARY = torch.library.Library("body_models", "FRAGMENT")
_LIBRARY.define("sparse_mm(Tensor rows, Tensor columns, Tensor values, Tensor inputs, int output_size) -> Tensor")


def _sparse_mm_impl(
    rows: Tensor,
    columns: Tensor,
    values: Tensor,
    inputs: Tensor,
    output_size: int,
) -> Tensor:
    indices = torch.stack((rows, columns))
    matrix = torch.sparse_coo_tensor(
        indices,
        values,
        (output_size, inputs.shape[0]),
    )
    return torch.sparse.mm(matrix, inputs)


def _sparse_mm_meta(
    rows: Tensor,
    columns: Tensor,
    values: Tensor,
    inputs: Tensor,
    output_size: int,
) -> Tensor:
    del rows, columns, values
    return inputs.new_empty((output_size, inputs.shape[1]))


_LIBRARY.impl("sparse_mm", _sparse_mm_impl, "CPU")
_LIBRARY.impl("sparse_mm", _sparse_mm_impl, "CUDA")
_LIBRARY.impl("sparse_mm", _sparse_mm_meta, "Meta")
_sparse_mm = cast(Callable[..., Tensor], torch.ops.body_models.sparse_mm)


class _SparseMM(torch.autograd.Function):
    """Make native sparse MM opaque to compilers and differentiable."""

    @staticmethod
    def forward(ctx, rows: Tensor, columns: Tensor, values: Tensor, inputs: Tensor, output_size: int) -> Tensor:
        ctx.save_for_backward(rows, columns, values)
        ctx.input_size = inputs.shape[0]
        return _sparse_mm(rows, columns, values, inputs, output_size)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[None, None, None, Tensor, None]:
        rows, columns, values = ctx.saved_tensors
        grad_inputs = _SparseMM.apply(
            columns,
            rows,
            values,
            grad_outputs[0],
            ctx.input_size,
        )
        return None, None, None, grad_inputs, None


class SparseLinear(nn.Module):
    """Sparse linear weights backed by the native sparse matrix multiply."""

    row_indices: Tensor
    column_indices: Tensor
    values: Tensor

    def __init__(self, weights: sparse_common.SparseMatrix) -> None:
        super().__init__()
        indices = torch.stack(
            (
                torch.as_tensor(weights.column_indices),
                torch.as_tensor(weights.row_indices),
            )
        )
        transpose = torch.sparse_coo_tensor(
            indices,
            torch.as_tensor(weights.values),
            (weights.shape[1], weights.shape[0]),
        ).coalesce()
        self.register_buffer("row_indices", transpose.indices()[0].clone(), persistent=True)
        self.register_buffer("column_indices", transpose.indices()[1].clone(), persistent=True)
        self.register_buffer("values", transpose.values().clone(), persistent=True)
        self.input_size, self.output_size = weights.shape

    def forward(
        self,
        inputs: Float[Tensor, "*batch input"],
    ) -> Float[Tensor, "*batch output"]:
        batch_shape = inputs.shape[:-1]
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        output = _SparseMM.apply(
            self.row_indices,
            self.column_indices,
            self.values,
            flat_inputs.T,
            self.output_size,
        ).T
        return output.reshape(*batch_shape, self.output_size)

    @property
    def shape(self) -> tuple[int, int]:
        return self.input_size, self.output_size

    def to_coo(self) -> sparse_common.SparseMatrix:
        return sparse_common.SparseMatrix(
            row_indices=self.column_indices,
            column_indices=self.row_indices,
            values=self.values,
            shape=self.shape,
        )


__all__ = ["SparseLinear"]
