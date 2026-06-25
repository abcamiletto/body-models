"""NumPy backend for the Unitree G1 rigid model."""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models.base import RigidBodyModel
from trimesh import Trimesh
from body_models.robots.g1.backends import core
from body_models.robots.g1.backends import numpy as backend
from body_models.robots.g1.io import load_model_data
from body_models.robots.g1.constants import G1_BODY_PRESETS, G1_JOINTS

__all__ = ["G1"]


class G1(RigidBodyModel):
    """Unitree G1 as rigid STL links attached to the Kimodo 34-joint skeleton."""

    JOINTS = G1_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: core.RotationType = "rotmat",
        convention: core.Convention = "soma",
    ) -> None:
        """Initialize the G1 model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            rotation_type: Rotation representation expected by pose inputs.
            convention: Skeleton convention used when loading rigid model data.
        """
        if rotation_type not in core.VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self.convention = convention
        self.weights = load_model_data(model_path, convention=convention)

    @property
    def faces(self) -> Int[np.ndarray, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.joint_names)

    @property
    def joint_names(self) -> list[str]:
        return self.weights.joint_names

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    @property
    def actuated_joint_names(self) -> list[str]:
        return self.weights.actuated_joint_names

    @property
    def num_actuated(self) -> int:
        return len(self.weights.actuated_joint_names)

    @property
    def actuated_joint_axes(self) -> Float[np.ndarray, "Q 3"]:
        return self.weights.actuated_joint_axes

    @property
    def actuated_joint_limits(self) -> Float[np.ndarray, "Q 2"]:
        return self.weights.actuated_joint_limits

    @property
    def link_names(self) -> list[str]:
        return self.weights.link_names

    @property
    def link_joint_indices(self) -> list[int]:
        return self.weights.link_joint_indices

    @property
    def link_vertex_starts(self) -> list[int]:
        return self.weights.link_vertex_starts

    @property
    def link_vertex_counts(self) -> list[int]:
        return self.weights.link_vertex_counts

    @property
    def link_face_starts(self) -> list[int]:
        return self.weights.link_face_starts

    @property
    def link_face_counts(self) -> list[int]:
        return self.weights.link_face_counts

    @property
    def link_geom_positions(self) -> Float[np.ndarray, "L 3"]:
        return self.weights.link_geom_positions

    @property
    def link_geom_rotations(self) -> Float[np.ndarray, "L 3 3"]:
        return self.weights.link_geom_rotations

    @property
    def num_vertices(self) -> int:
        return self.weights.vertices.shape[0]

    def forward_skeleton(
        self,
        body_pose: Float[np.ndarray, "B Q N"] | Float[np.ndarray, "B Q 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[np.ndarray, "B J 4 4"]:
        """Compute posed joint transforms.

        Args:
            body_pose: Local body joint rotations.
            global_translation: Global model translation.
            global_rotation: Global model rotation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
        return backend.forward_skeleton(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def forward_meshes(
        self,
        body_pose: Float[np.ndarray, "B Q N"] | Float[np.ndarray, "B Q 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    ) -> list[Trimesh]:
        """Compute posed model meshes.

        Args:
            body_pose: Local body joint rotations.
            global_translation: Global model translation.
            global_rotation: Global model rotation.

        Returns:
            One posed model mesh per batch element.
        """
        return backend.forward_meshes(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
            rotation_type=self.rotation_type,
        )

    def forward_links(
        self,
        body_pose: Float[np.ndarray, "B Q N"] | Float[np.ndarray, "B Q 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    ) -> Float[np.ndarray, "B L 4 4"]:
        return backend.forward_links(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype=np.float32) -> dict[str, np.ndarray]:
        pose_ref = np.zeros((*batch_dims, self.num_actuated, 3), dtype=dtype)
        global_ref = np.zeros((*batch_dims, 3), dtype=dtype)
        body_pose = SO3.identity_as(
            pose_ref,
            batch_dims=(*batch_dims, self.num_actuated),
            rotation_type=self.rotation_type,
            xp=np,
        ).copy()
        global_rotation = SO3.identity_as(
            global_ref,
            batch_dims=batch_dims,
            rotation_type=core.GLOBAL_ROTATION_TYPES[self.rotation_type],
            xp=np,
        ).copy()
        return {
            "body_pose": body_pose,
            "global_rotation": global_rotation,
            "global_translation": np.zeros((*batch_dims, 3), dtype=dtype),
        }

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        axis_angle = np.asarray(G1_BODY_PRESETS["t_pose"], dtype=params["body_pose"].dtype)
        axis_angle = np.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        dst_kwargs = {"hinge": {"axes": self.actuated_joint_axes}}.get(self.rotation_type, {})
        params["body_pose"] = SO3.convert(
            axis_angle,
            src="axis_angle",
            dst=self.rotation_type,
            dst_kwargs=dst_kwargs,
            xp=np,
        ).copy()
        return params

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        axis_angle = np.asarray(G1_BODY_PRESETS["a_pose"], dtype=params["body_pose"].dtype)
        axis_angle = np.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        dst_kwargs = {"hinge": {"axes": self.actuated_joint_axes}}.get(self.rotation_type, {})
        params["body_pose"] = SO3.convert(
            axis_angle,
            src="axis_angle",
            dst=self.rotation_type,
            dst_kwargs=dst_kwargs,
            xp=np,
        ).copy()
        return params
