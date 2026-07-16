from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, NotRequired, TypedDict

from array_api_compat import get_namespace
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models.common import eye_as, zeros_as
from body_models.constants import Joint
from trimesh import Trimesh

Array = Any


class SkinningPayload(TypedDict):
    """Renderer-ready linear blend skinning inputs."""

    rest_vertices: Float[Array, "*batch V 3"]
    skinning_transforms: Float[Array, "*batch J 4 4"]
    pose_offsets: NotRequired[Float[Array, "*batch V 3"]]
    skin_weights: Float[Array, "V J"]
    faces: Int[Array, "F C"]


class _ArticulatedModel(ABC):
    """Shared skeleton interface for skinned and rigid articulated models."""

    parents: list[int]
    has_hands: bool = False
    has_head: bool = False
    skinning_backends: ClassVar[tuple[str, ...]] = ("numpy",)
    JOINTS: ClassVar[Mapping[Joint, str]] = {}

    @property
    @abstractmethod
    def faces(self) -> Int[Array, "F C"]:
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
    def forward_skeleton(self, *args, **kwargs) -> Float[Array, "*batch J 4 4"]:
        """
        Compute skeleton joint transforms.

        Signature varies by model. Outputs use the model's native coordinate system.
        in meters.

        Returns:
            World-space 4x4 transformation matrices [B, J, 4, 4] in meters.
        """

    @abstractmethod
    def get_rest_pose(self, batch_dims: tuple[int, ...] = ()) -> dict[str, Float[Array, "..."]]:
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
    ) -> dict[str, Float[Array, "..."]]:
        """Get parameters for the SMPL-style T-pose."""
        raise NotImplementedError("Canonical body poses are not defined for this model.")

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> dict[str, Float[Array, "..."]]:
        """Get parameters for the MHR-style A-pose."""
        raise NotImplementedError("Canonical body poses are not defined for this model.")


class SkinnedModel(_ArticulatedModel):
    """Base class for models that expose one skinned mesh."""

    @property
    @abstractmethod
    def skin_weights(self) -> Float[Array, "V J"]:
        """Skinning weights mapping vertices to joints. Shape [V, J]."""

    @property
    @abstractmethod
    def rest_vertices(self) -> Float[Array, "V 3"]:
        """Mesh vertices in rest pose. Shape [V, 3]."""

    @abstractmethod
    def forward_vertices(self, *args, **kwargs) -> Float[Array, "*batch V 3"]:
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

    @staticmethod
    def _validate_identity_arguments(identity: Any | None, **raw_parameters: Any | None) -> None:
        if identity is None:
            return
        conflicts = [name for name, value in raw_parameters.items() if value is not None]
        if conflicts:
            names = ", ".join(conflicts)
            raise ValueError(f"identity cannot be combined with raw identity parameters: {names}")


class RigidBodyModel(_ArticulatedModel):
    """Base class for rigid articulated models."""

    mujoco_to_model: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    @property
    @abstractmethod
    def actuated_joint_names(self) -> list[str]:
        """Actuated pose coordinate names in ``body_pose``/``hand_pose`` order."""

    @property
    def num_actuated(self) -> int:
        """Number of actuated pose coordinates."""
        return len(self.actuated_joint_names)

    @property
    def actuated_joint_slices(self) -> Mapping[str, slice]:
        """Consecutive scalar coordinate slices keyed by actuated joint name."""
        slices = {}
        seen = set()
        start = 0
        names = self.actuated_joint_names
        while start < len(names):
            name = names[start]
            if name in seen:
                raise ValueError(f"Actuated joint name {name!r} is repeated in non-consecutive groups.")
            seen.add(name)
            stop = start + 1
            while stop < len(names) and names[stop] == name:
                stop += 1
            slices[name] = slice(start, stop)
            start = stop
        return slices

    def unpack_pose(self, pose: Float[Array, "*batch Q"]) -> dict[str, Float[Array, "*batch dof"]]:
        """Unpack a flattened pose ``[..., Q]`` into ``name -> [..., dof]`` arrays."""
        if pose.shape[-1] != self.num_actuated:
            raise ValueError(f"pose must have shape [..., {self.num_actuated}], got {tuple(pose.shape)}")
        return {name: pose[..., joint_slice] for name, joint_slice in self.actuated_joint_slices.items()}

    def pack_pose(self, pose_by_joint: Mapping[str, Float[Array, "*batch dof"]]) -> Float[Array, "*batch Q"]:
        """Pack ``name -> [..., dof]`` arrays into a flattened pose ``[..., Q]``."""
        pieces = []
        expected_names = set(self.actuated_joint_slices)
        extra_names = set(pose_by_joint) - expected_names
        if extra_names:
            raise KeyError(f"Unknown actuated joint names: {sorted(extra_names)}")
        for name, joint_slice in self.actuated_joint_slices.items():
            if name not in pose_by_joint:
                raise KeyError(f"Missing actuated joint name: {name!r}")
            value = pose_by_joint[name]
            dof = joint_slice.stop - joint_slice.start
            if value.shape[-1] != dof:
                raise ValueError(f"{name!r} must have shape [..., {dof}], got {tuple(value.shape)}")
            pieces.append(value)
        return get_namespace(*pieces).concat(pieces, axis=-1)

    def to_qpos(
        self,
        body_pose: Float[Array, "*batch Q"],
        global_translation: Float[Array, "*batch 3"] | None = None,
        *,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        clamp_to_limits: bool = False,
    ) -> Float[Array, "*batch qpos"]:
        """Build full MuJoCo ``qpos`` as ``[root_xyz, root_wxyz, body_pose]``.

        ``body_pose`` is the model's flattened scalar coordinate vector ``[..., Q]``.
        The root prefix is converted from the model coordinate frame to MuJoCo's
        coordinate frame.
        """
        if body_pose.shape[-1] != self.num_actuated:
            raise ValueError(f"body_pose must have shape [..., {self.num_actuated}], got {tuple(body_pose.shape)}")

        xp = get_namespace(body_pose)
        batch_shape = tuple(body_pose.shape[:-1])
        if global_translation is None:
            global_translation = zeros_as(body_pose, shape=(*batch_shape, 3), xp=xp)
        if global_rotation is None:
            root_ref = zeros_as(body_pose, shape=(*batch_shape, 3), xp=xp)
            root_rot = eye_as(root_ref, batch_dims=batch_shape, xp=xp)
        else:
            root_rot = SO3.convert(global_rotation, src="axis_angle", dst="rotmat", xp=xp)

        coord = xp.asarray(self.mujoco_to_model, dtype=body_pose.dtype)
        model_to_mujoco = coord.mT if hasattr(coord, "mT") else xp.swapaxes(coord, -1, -2)
        root_t = xp.squeeze(model_to_mujoco @ global_translation[..., None], axis=-1)
        root_rot_mujoco = model_to_mujoco @ root_rot @ coord
        root_quat = SO3.conversions.from_rotmat_to_quat(root_rot_mujoco, convention="wxyz", xp=xp)

        if clamp_to_limits:
            limits = xp.asarray(self.actuated_joint_limits, dtype=body_pose.dtype)
            body_pose = xp.clip(body_pose, limits[:, 0], limits[:, 1])
        return xp.concat([root_t, root_quat, body_pose], axis=-1)

    @property
    @abstractmethod
    def actuated_joint_limits(self) -> Float[Array, "Q 2"]:
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
    def forward_links(self, *args, **kwargs) -> Float[Array, "*batch L 4 4"]:
        """Compute world-space 4x4 link transforms as the array/autograd primitive."""

    @abstractmethod
    def forward_meshes(self, *args, **kwargs) -> Sequence[Trimesh]:
        """Build one renderer-facing mesh per batch element from link transforms."""
