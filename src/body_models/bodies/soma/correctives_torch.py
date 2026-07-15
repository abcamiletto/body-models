"""Torch sparse lowering of SOMA's corrective network."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from body_models.bodies.soma.correctives import hidden_activations
from body_models.runtime import TorchRuntime


class TorchCorrectiveNetwork(nn.Module):
    """SOMA corrective network with sparse and compile-safe output paths."""

    def __init__(self, data: Any) -> None:
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
        self._runtime = TorchRuntime()

    def forward(self, data: Any, pose_rotations: Any) -> Any:
        hidden = hidden_activations(self._runtime, data, pose_rotations)
        batch_shape = hidden.shape[:-1]
        if torch.compiler.is_compiling():
            correctives = data.correctives
            contributions = hidden[..., correctives.corrective_W2_rows]
            contributions = contributions * correctives.corrective_W2_values
            output_size = data.mean_full.shape[0] * 3
            output = torch.zeros(
                (*batch_shape, output_size),
                dtype=hidden.dtype,
                device=hidden.device,
            )
            indices = torch.broadcast_to(correctives.corrective_W2_cols, contributions.shape)
            output = output.scatter_add(-1, indices, contributions)
        else:
            output = torch.sparse.mm(self.transpose, hidden.reshape(-1, hidden.shape[-1]).T).T
        return output.reshape(*batch_shape, data.mean_full.shape[0], 3)


__all__ = ["TorchCorrectiveNetwork"]
