from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from nanomanifold import SO3

from body_models.constants import Joint


class BodyModel(ABC):
    """Abstract base class for body models.

    Framework-agnostic interface that all backends (torch, numpy, jax) implement.
    Array types are left as Any to support different frameworks.

    PyTorch backends should inherit from both BodyModel and nn.Module:
        class SMPL(BodyModel, nn.Module):
            ...
    """

    parents: list[int]
    # True for models whose meshes are rigidly attached to bodies (no LBS skin
    # weights), e.g. G1 and MyoFullBody. Used to gate skin_weights / viser
    # skinned-mesh exports without sniffing for NotImplementedError.
    is_rigid_body: bool = False
    # True for models that expose MJCF-style muscle via-points and tendons
    # (currently MyoFullBody only). Renderers branch on this to draw muscles.
    has_tendons: bool = False
    kernels: ClassVar[tuple[str, ...]] = ("numpy",)
    JOINTS: ClassVar[Mapping[Joint, str]] = {}

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
    def joint_names(self) -> list[str]:
        """Joint names in joint index order."""

    @property
    def _standard_joints(self) -> Mapping[Joint, str]:
        return self.JOINTS

    def joint_index(self, joint: Joint) -> int:
        """Resolve a standard joint to this model's native joint index."""
        if not isinstance(joint, Joint):
            raise TypeError("joint_index() expects a body_models.Joint; use joint_names.index(...) for native names.")
        try:
            native_name = self._standard_joints[joint]
        except KeyError as exc:
            raise KeyError(f"{self.__class__.__name__} has no standard joint {joint.value!r}") from exc
        return self.joint_names.index(native_name)

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

        Signature varies by model. Outputs use the model's native coordinate system.
        in meters.

        Returns:
            Mesh vertices [B, V, 3] in meters.
        """

    @abstractmethod
    def forward_skeleton(self, *args, **kwargs) -> Any:
        """
        Compute skeleton joint transforms.

        Signature varies by model. Outputs use the model's native coordinate system.
        in meters.

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

    def to_viser_bones(self, **forward_kwargs: Any) -> dict[str, np.ndarray]:
        """Export parent-relative bone poses for ``viser`` from ``forward_skeleton()`` kwargs."""
        if not forward_kwargs:
            forward_kwargs = self.get_rest_pose(batch_size=1)
        if "joint_indices" in forward_kwargs:
            raise ValueError("to_viser_bones() requires the full skeleton; do not pass joint_indices.")

        skeleton = np.asarray(self.forward_skeleton(**forward_kwargs))
        if skeleton.shape[0] != 1:
            raise ValueError(f"to_viser_bones() expects batch size 1, got {skeleton.shape[0]}")
        if len(self.parents) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joint parents, got {len(self.parents)}")

        world = skeleton[0]
        local = world.copy()
        for joint_index, parent_index in enumerate(self.parents):
            if parent_index >= 0:
                local[joint_index] = np.linalg.solve(world[parent_index], world[joint_index])

        bone_wxyzs = SO3.conversions.from_rotmat_to_quat(local[:, :3, :3], convention="wxyz", xp=np)
        bone_positions = local[:, :3, 3]
        return {"bone_wxyzs": bone_wxyzs, "bone_positions": bone_positions.copy()}

    def to_viser_skinned_mesh(self, **forward_kwargs: Any) -> dict[str, np.ndarray]:
        """Export bind-pose mesh data for ``viser`` from ``forward_vertices()`` / ``forward_skeleton()`` kwargs."""
        if not forward_kwargs:
            forward_kwargs = self.get_rest_pose(batch_size=1)
        if "vertex_indices" in forward_kwargs:
            raise ValueError("to_viser_skinned_mesh() requires the full mesh; do not pass vertex_indices.")
        if "joint_indices" in forward_kwargs:
            raise ValueError("to_viser_skinned_mesh() requires the full skeleton; do not pass joint_indices.")

        vertices = np.asarray(self.forward_vertices(**forward_kwargs))
        if vertices.shape[0] != 1:
            raise ValueError(f"to_viser_skinned_mesh() expects batch size 1, got {vertices.shape[0]}")

        faces = np.asarray(self.faces)
        if faces.shape[1] == 4:
            faces = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)
        elif faces.shape[1] != 3:
            raise ValueError(f"Expected triangular or quad faces, got shape {faces.shape}")

        return {
            "vertices": vertices[0].copy(),
            "faces": faces.copy(),
            "skin_weights": np.asarray(self.skin_weights).copy(),
            **self.to_viser_bones(**forward_kwargs),
        }
