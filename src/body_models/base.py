from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, NotRequired, TypedDict

from body_models.constants import Joint
from trimesh import Trimesh


class SkinningPayload(TypedDict):
    """Renderer-ready linear blend skinning inputs."""

    rest_vertices: Any
    skinning_transforms: Any
    pose_offsets: NotRequired[Any]
    skin_weights: Any
    faces: Any


class BodyModel(ABC):
    """Shared skeleton and pose interface for body-like models.

    Framework-agnostic interface that all backends (torch, numpy, jax) implement.
    Array types are left as Any to support different frameworks.
    """

    parents: list[int]
    # True for models that expose a hand_pose parameter.
    has_hands: bool = False
    # True for models that expose a head_pose or head_rotation parameter.
    has_head: bool = False
    kernels: ClassVar[tuple[str, ...]] = ("numpy",)
    JOINTS: ClassVar[Mapping[Joint, str]] = {}
    POSE_PARAMETER_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "body_pose",
            "hand_pose",
            "head_pose",
            "pelvis_rotation",
            "head_rotation",
            "wrist_rotation",
            "global_rotation",
            "global_translation",
        }
    )

    @property
    @abstractmethod
    def faces(self) -> Any:
        """Mesh face indices. Shape [F, 3] for triangles or [F, 4] for quads."""

    @property
    def num_joints(self) -> int:
        """Number of joints in the skeleton."""
        return len(self.joint_names)

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

    def get_bind_params(self, **forward_kwargs: Any) -> dict[str, Any]:
        """Return rest-pose params with identity/shape params copied from current params."""
        if not forward_kwargs:
            return self.get_rest_pose()

        bind_params = self.get_rest_pose()
        for name, value in forward_kwargs.items():
            if name in bind_params and name not in self.POSE_PARAMETER_NAMES:
                bind_params[name] = value
        return bind_params


class SkinnedModel(BodyModel):
    """Base class for models that expose one skinned mesh."""

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

    def prepare_skinning(self, *, identity: Mapping[str, Any], pose: Mapping[str, Any]) -> SkinningPayload:
        """Pack prepared model state into renderer-ready skinning inputs."""
        skinning: SkinningPayload = {
            "rest_vertices": identity["rest_vertices"],
            "skinning_transforms": pose["skinning_transforms"],
            "skin_weights": self.skin_weights,
            "faces": self.faces,
        }
        if "pose_offsets" in pose:
            skinning["pose_offsets"] = pose["pose_offsets"]
        return skinning


class RigidBodyModel(BodyModel):
    """Base class for rigid articulated models."""

    @property
    @abstractmethod
    def actuated_joint_names(self) -> list[str]:
        """Actuated pose coordinate names in ``body_pose``/``hand_pose`` order."""

    @property
    def num_actuated(self) -> int:
        """Number of actuated pose coordinates."""
        return len(self.actuated_joint_names)

    @property
    @abstractmethod
    def actuated_joint_limits(self) -> Any:
        """Limits for each actuated pose coordinate. Shape [Q, 2]."""

    @property
    @abstractmethod
    def actuated_joint_types(self) -> list[str]:
        """Actuated pose coordinate types in ``actuated_joint_names`` order."""

    @property
    @abstractmethod
    def link_names(self) -> list[str]:
        """Link mesh names in link index order."""

    @property
    @abstractmethod
    def link_joint_indices(self) -> list[int]:
        """Joint index associated with each link mesh."""

    @abstractmethod
    def forward_links(self, *args, **kwargs) -> Any:
        """Compute world-space 4x4 link transforms as the array/autograd primitive."""

    @abstractmethod
    def forward_meshes(self, *args, **kwargs) -> Sequence[Trimesh]:
        """Build one renderer-facing mesh per batch element from link transforms."""
