"""Torch sparse lowering of SOMA's corrective network."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from body_models.runtime import ArrayRuntime


class TorchCorrectiveNetwork(nn.Module):
    """SOMA corrective network with sparse and compile-safe output paths."""

    rows: Int[Tensor, "K"]
    cols: Int[Tensor, "K"]
    values: Float[Tensor, "K"]

    def __init__(self, runtime: ArrayRuntime, data: Any) -> None:
        super().__init__()
        correctives = data.correctives
        indices = torch.stack((correctives.corrective_W2_cols, correctives.corrective_W2_rows))
        output_size = data.mean_full.shape[0] * 3
        hidden_size = correctives.corrective_W1.shape[1]
        transpose = torch.sparse_coo_tensor(
            indices,
            correctives.corrective_W2_values,
            (output_size, hidden_size),
        ).coalesce()
        self.register_buffer("transpose", transpose, persistent=False)
        self.register_buffer("rows", correctives.corrective_W2_rows, persistent=False)
        self.register_buffer("cols", correctives.corrective_W2_cols, persistent=False)
        self.register_buffer("values", correctives.corrective_W2_values, persistent=False)
        self._num_vertices = data.mean_full.shape[0]
        self._runtime = runtime

    def forward(
        self,
        hidden: Float[Tensor, "*batch H"],
    ) -> Float[Tensor, "*batch Vf 3"]:
        batch_shape = hidden.shape[:-1]
        if torch.compiler.is_compiling():
            contributions = hidden[..., self.rows] * self.values
            output_size = self._num_vertices * 3
            output = torch.zeros(
                (*batch_shape, output_size),
                dtype=hidden.dtype,
                device=hidden.device,
            )
            indices = torch.broadcast_to(self.cols, contributions.shape)
            output = output.scatter_add(-1, indices, contributions)
        else:
            output = torch.sparse.mm(self.transpose, hidden.reshape(-1, hidden.shape[-1]).T).T
        return output.reshape(*batch_shape, self._num_vertices, 3)


__all__ = ["TorchCorrectiveNetwork"]
