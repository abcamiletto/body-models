"""PyTorch backend for the GarmentMeasurements PCA body model."""

from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from ..base import BodyModel
from ..rotations import VALID_ROTATION_TYPES
from . import core
from .io import load_model_data

__all__ = ["GarmentMeasurements"]


class GarmentMeasurements(BodyModel, nn.Module):
    """GarmentMeasurements PCA body model with PyTorch backend."""

    mean_vertices: Float[Tensor, "V 3"]
    components: Float[Tensor, "V 3 C"]
    eigenvalues: Float[Tensor, "C"]
    _skin_weights: Float[Tensor, "V 1"]

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: core.RotationType = "axis_angle",
    ) -> None:
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        super().__init__()

        data = load_model_data(model_path=model_path, dtype="float32")
        for key in ["mean_vertices", "components", "eigenvalues"]:
            self.register_buffer(key, torch.as_tensor(data[key], dtype=torch.float32), persistent=False)

        self.register_buffer(
            "_skin_weights",
            torch.ones((self.mean_vertices.shape[0], 1), dtype=torch.float32),
            persistent=False,
        )
        self._faces = torch.as_tensor(data["faces"], dtype=torch.int64)
        self._joint_names = data["joint_names"]
        self.parents = data["parents"]
        self.rotation_type = rotation_type

    @property
    def faces(self) -> Int[Tensor, "F _"]:
        return self._faces

    @property
    def num_joints(self) -> int:
        return 1

    @property
    def num_vertices(self) -> int:
        return self.mean_vertices.shape[0]

    @property
    def num_shape_components(self) -> int:
        return self.eigenvalues.shape[0]

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def skin_weights(self) -> Float[Tensor, "V J"]:
        return self._skin_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.mean_vertices

    def forward_vertices(
        self,
        shape: Float[Tensor, "B C"],
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        vertex_indices: list[int] | None = None,
    ) -> Float[Tensor, "B V 3"]:
        return core.forward_vertices(
            mean_vertices=self.mean_vertices,
            components=self.components,
            eigenvalues=self.eigenvalues,
            shape=shape,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
            xp=torch,
        )

    def forward_skeleton(
        self,
        shape: Float[Tensor, "B C"],
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[Tensor, "B J 4 4"]:
        return core.forward_skeleton(
            shape=shape,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
            xp=torch,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype: torch.dtype | None = None) -> dict[str, Tensor]:
        dtype = dtype or self.mean_vertices.dtype
        device = self.mean_vertices.device
        zeros = torch.zeros((batch_size,), dtype=dtype, device=device)
        return {
            "shape": torch.zeros((batch_size, self.num_shape_components), dtype=dtype, device=device),
            "global_rotation": SO3.identity_as(
                zeros,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "global_translation": torch.zeros((batch_size, 3), dtype=dtype, device=device),
        }
