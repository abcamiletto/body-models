"""NumPy backend for the SMPL humanoid robot."""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3
from trimesh import Trimesh

from body_models.base import RigidBodyModel
from body_models.robots.smpl_humanoid.backends import numpy as backend
from body_models.robots.smpl_humanoid.constants import BODY_JOINTS, SMPL_BODY_PRESETS, SMPL_HUMANOID_JOINTS
from body_models.robots.smpl_humanoid.io import load_model_data

__all__ = ["SmplHumanoid"]


class SmplHumanoid(RigidBodyModel):
    """Rigid humanoid loaded from MJCF using the canonical 24-joint SMPL hierarchy."""

    JOINTS = SMPL_HUMANOID_JOINTS

    def __init__(
        self,
        source: Path | str = "humenv",
    ) -> None:
        self.weights = load_model_data(source)

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
    def actuated_joint_limits(self) -> Float[np.ndarray, "Q 2"]:
        return self.weights.actuated_joint_limits

    @property
    def actuated_joint_types(self) -> list[str]:
        return self.weights.actuated_joint_types

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
        body_pose: Float[np.ndarray, "B Q"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[np.ndarray, "B 24 4 4"]:
        return backend.forward_skeleton(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
        )

    def forward_meshes(
        self,
        body_pose: Float[np.ndarray, "B Q"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
    ) -> list[Trimesh]:
        return backend.forward_meshes(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
        )

    def forward_links(
        self,
        body_pose: Float[np.ndarray, "B Q"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
    ) -> Float[np.ndarray, "B 24 4 4"]:
        return backend.forward_links(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
        )

    def to_qpos(
        self,
        pose: Float[np.ndarray, "B Q"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
        clamp_to_limits: bool = False,
    ) -> Float[np.ndarray, "B 76"]:
        axis_angle = pose.reshape(*pose.shape[:-1], len(BODY_JOINTS), 3)
        euler = SO3.conversions.from_axis_angle_to_euler(axis_angle, convention="XYZ", xp=np)
        return super().to_qpos(
            euler.reshape(*pose.shape),
            global_translation,
            global_rotation=global_rotation,
            clamp_to_limits=clamp_to_limits,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype=np.float32) -> dict[str, np.ndarray]:
        return {
            "body_pose": np.zeros((*batch_dims, self.num_actuated), dtype=dtype),
            "global_rotation": np.zeros((*batch_dims, 3), dtype=dtype),
            "global_translation": np.zeros((*batch_dims, 3), dtype=dtype),
        }

    def get_tpose(self, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, np.ndarray]:
        return self._preset_pose("t_pose", batch_dims=batch_dims, **kwargs)

    def get_apose(self, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, np.ndarray]:
        return self._preset_pose("a_pose", batch_dims=batch_dims, **kwargs)

    def _preset_pose(self, name: str, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        axis_angle = np.asarray(SMPL_BODY_PRESETS[name], dtype=params["body_pose"].dtype)
        ordered = np.concatenate([axis_angle[smpl_index] for _, smpl_index in BODY_JOINTS])
        params["body_pose"] = np.broadcast_to(ordered, (*batch_dims, ordered.shape[0])).copy()
        return params
