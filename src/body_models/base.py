from abc import ABC, abstractmethod
from typing import Any


class BodyModel(ABC):
    """Abstract base class for body models.

    Framework-agnostic interface that all backends (torch, numpy, jax) implement.
    Array types are left as Any to support different frameworks.

    PyTorch backends should inherit from both BodyModel and nn.Module:
        class SMPL(BodyModel, nn.Module):
            ...
    """

    @property
    @abstractmethod
    def faces(self) -> Any:
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
    def skin_weights(self) -> Any:
        """Skinning weights mapping vertices to joints. Shape [V, J]."""

    @property
    @abstractmethod
    def rest_vertices(self) -> Any:
        """Mesh vertices in rest pose. Shape [V, 3]."""

    @abstractmethod
    def forward_vertices(self, *args, **kwargs) -> Any:
        """
        Compute mesh vertices.

        Signature varies by model. Outputs are in Y-up coordinate system
        with feet at floor level (Y=0), in meters.

        Returns:
            Mesh vertices [B, V, 3] in meters.
        """

    @abstractmethod
    def forward_skeleton(self, *args, **kwargs) -> Any:
        """
        Compute skeleton joint transforms.

        Signature varies by model. Outputs are in Y-up coordinate system
        with feet at floor level (Y=0), in meters.

        Returns:
            World-space 4x4 transformation matrices [B, J, 4, 4] in meters.
        """

    @abstractmethod
    def get_rest_pose(self, batch_size: int = 1) -> dict[str, Any]:
        """
        Get default rest pose parameters for this model.

        Args:
            batch_size: Number of instances in the batch.

        Returns:
            Dictionary with model-specific parameter keys. All arrays are
            zero-initialized or set to identity poses.
        """
