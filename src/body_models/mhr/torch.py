"""PyTorch backend for MHR model."""

from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from body_models import common
from body_models.base import BodyModel
from body_models.mhr.backends import torch as backend
from body_models.mhr.io import get_model_path, load_model_data

__all__ = ["MHR"]


class MHR(BodyModel, nn.Module):
    """MHR body model with PyTorch backend."""

    SHAPE_DIM = 45
    EXPR_DIM = 72

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        lod: int = 1,
        simplify: float = 1.0,
    ) -> None:
        super().__init__()
        self.weights = common.torchify(load_model_data(get_model_path(model_path), lod=lod, simplify=simplify))

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.parents)

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self.weights.base_vertices.shape[0]

    @property
    def pose_dim(self) -> int:
        return self.weights.parameter_transform.shape[1] - self.SHAPE_DIM

    @property
    def skin_weights(self) -> Float[Tensor, "V J"]:
        dense = torch.zeros(
            self.weights.skin_weights.shape[0],
            self.num_joints,
            device=self.weights.skin_weights.device,
            dtype=self.weights.skin_weights.dtype,
        )
        dense.scatter_(1, self.weights.skin_indices, self.weights.skin_weights)
        return dense

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.weights.base_vertices * 0.01

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        shape: Float[Tensor, "B|1 45"],
        pose: Float[Tensor, "B 204"],
        expression: Float[Tensor, "B 72"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[Tensor, "B V 3"]:
        return backend.forward_vertices(
            weights=self.weights,
            shape=shape,
            pose=pose,
            expression=expression,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
        )

    def forward_skeleton(
        self,
        shape: Float[Tensor, "B|1 45"],
        pose: Float[Tensor, "B 204"],
        expression: Float[Tensor, "B 72"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[Tensor, "B J 4 4"]:
        return backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            pose=pose,
            expression=expression,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        device = self.rest_vertices.device
        return {
            "shape": torch.zeros((1, self.SHAPE_DIM), device=device, dtype=dtype),
            "pose": torch.zeros((batch_size, self.pose_dim), device=device, dtype=dtype),
            "expression": torch.zeros((batch_size, self.EXPR_DIM), device=device, dtype=dtype),
            "global_rotation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
            "global_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }
