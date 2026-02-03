from abc import ABC, abstractmethod

import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class BodyModel(ABC, nn.Module):
    @property
    @abstractmethod
    def faces(self) -> Int[Tensor, "F K"]:
        """Mesh face indices. Shape [F, 3] for triangles or [F, 4] for quads."""

    @property
    @abstractmethod
    def num_joints(self) -> int:
        """Number of joints in the skeleton."""

    @property
    @abstractmethod
    def num_vertices(self) -> int:
        """Number of mesh vertices."""

    @property
    @abstractmethod
    def skin_weights(self) -> Float[Tensor, "V J"]:
        """Skinning weights mapping vertices to joints. Shape [V, J]."""

    @property
    @abstractmethod
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        """Mesh vertices in rest pose. Shape [V, 3]."""

    @abstractmethod
    def forward_vertices(self, *args, **kwargs) -> Float[Tensor, "B V 3"]:
        """
        Compute mesh vertices.

        Signature varies by model. Outputs are in Y-up coordinate system
        with feet at floor level (Y=0), in meters.

        Returns:
            Mesh vertices [B, V, 3] in meters.
        """

    @abstractmethod
    def forward_skeleton(self, *args, **kwargs) -> Float[Tensor, "B J 4 4"]:
        """
        Compute skeleton joint transforms.

        Signature varies by model. Outputs are in Y-up coordinate system
        with feet at floor level (Y=0), in meters.

        Returns:
            World-space 4x4 transformation matrices [B, J, 4, 4] in meters.
        """

    @abstractmethod
    def get_rest_pose(self, batch_size: int = 1) -> dict[str, Tensor]:
        """
        Get default rest pose parameters for this model.

        Args:
            batch_size: Number of instances in the batch.

        Returns:
            Dictionary with model-specific parameter keys. All tensors are
            zero-initialized or set to identity poses.
        """
