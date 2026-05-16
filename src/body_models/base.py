from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, TypedDict

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models.constants import Joint


class ViserBones(TypedDict):
    bone_wxyzs: Float[np.ndarray, "J 4"]
    bone_positions: Float[np.ndarray, "J 3"]


class ViserSkinnedMesh(TypedDict):
    vertices: Float[np.ndarray, "V 3"]
    faces: Int[np.ndarray, "F 3"]
    skin_weights: Float[np.ndarray, "V J"]
    bone_wxyzs: Float[np.ndarray, "J 4"]
    bone_positions: Float[np.ndarray, "J 3"]


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
    # True for models that expose a hand_pose parameter and accept hands presets.
    has_hands: bool = False
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
    def common_joints(self) -> Mapping[Joint, str]:
        """Common anatomical joints mapped to this model's native joint names."""
        return self.JOINTS

    def joint_index(self, joint: Joint) -> int:
        """Resolve a standard joint to this model's native joint index."""
        if not isinstance(joint, Joint):
            raise TypeError("joint_index() expects a body_models.Joint; use joint_names.index(...) for native names.")
        try:
            native_name = self.common_joints[joint]
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
    def get_rest_pose(self, batch_dims: tuple[int, ...] = ()) -> dict[str, Any]:
        """
        Get default rest pose parameters for this model.

        Args:
            batch_dims: Leading batch dimensions.

        Returns:
            Dictionary with model-specific parameter keys. All arrays are
            zero-initialized or set to identity poses.
        """

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get parameters for the SMPL-style T-pose."""
        raise NotImplementedError("Canonical body poses are not defined for this model.")

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get parameters for the MHR-style A-pose."""
        raise NotImplementedError("Canonical body poses are not defined for this model.")

    def to_viser_bones(self, **forward_kwargs: Any) -> ViserBones:
        """Export world-space bone poses for ``viser`` from ``forward_skeleton()`` kwargs."""
        if not forward_kwargs:
            forward_kwargs = self.get_rest_pose()
        if "joint_indices" in forward_kwargs:
            raise ValueError("to_viser_bones() requires the full skeleton; do not pass joint_indices.")

        skeleton = np.asarray(self.forward_skeleton(**forward_kwargs))
        if skeleton.ndim != 3 or skeleton.shape[-2:] != (4, 4):
            raise ValueError(f"to_viser_bones() expects unbatched skeleton shape (N, 4, 4), got {skeleton.shape}")
        world = skeleton
        bone_wxyzs = SO3.conversions.from_rotmat_to_quat(world[:, :3, :3], convention="wxyz", xp=np)
        bone_positions = world[:, :3, 3]
        return {"bone_wxyzs": bone_wxyzs, "bone_positions": bone_positions.copy()}

    def to_viser_skinned_mesh(self, **forward_kwargs: Any) -> ViserSkinnedMesh:
        """Export bind-pose mesh data for ``viser`` from ``forward_vertices()`` / ``forward_skeleton()`` kwargs."""
        if not forward_kwargs:
            forward_kwargs = self.get_rest_pose()
        if "vertex_indices" in forward_kwargs:
            raise ValueError("to_viser_skinned_mesh() requires the full mesh; do not pass vertex_indices.")
        if "joint_indices" in forward_kwargs:
            raise ValueError("to_viser_skinned_mesh() requires the full skeleton; do not pass joint_indices.")

        vertices = np.asarray(self.forward_vertices(**forward_kwargs))
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise ValueError(f"to_viser_skinned_mesh() expects unbatched vertices shape (N, 3), got {vertices.shape}")

        faces = np.asarray(self.faces)
        if faces.shape[1] == 4:
            faces = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)
        elif faces.shape[1] != 3:
            raise ValueError(f"Expected triangular or quad faces, got shape {faces.shape}")

        bones = self.to_viser_bones(**forward_kwargs)
        return {
            "vertices": vertices.copy(),
            "faces": faces.copy(),
            "skin_weights": _viser_skin_weights(np.asarray(self.skin_weights)),
            "bone_wxyzs": bones["bone_wxyzs"],
            "bone_positions": bones["bone_positions"],
        }


def _viser_skin_weights(skin_weights: Float[np.ndarray, "V J"]) -> Float[np.ndarray, "V J"]:
    """Return skin weights in the 4-influence format used by viser."""
    weights = np.asarray(skin_weights).copy()
    pruned_indices = np.argsort(weights, axis=-1)[:, :-4]
    weights[np.arange(weights.shape[0])[:, None], pruned_indices] = 0.0
    row_sums = weights.sum(axis=-1, keepdims=True)
    return np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums > 0)
