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

    def from_smpl_motion(
        self,
        smpl_body_pose: Float[np.ndarray, "B 23 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
        pelvis_rotation: Float[np.ndarray, "B 3"] | None = None,
    ) -> dict[str, np.ndarray]:
        ordered = np.stack([smpl_body_pose[..., smpl_index, :] for _, smpl_index in BODY_JOINTS], axis=-2)
        motion = {
            "body_pose": SO3.conversions.from_axis_angle_to_euler(ordered, convention="XYZ", xp=np).reshape(
                *smpl_body_pose.shape[:-2], self.num_actuated
            )
        }
        if global_translation is not None:
            motion["global_translation"] = global_translation
        if global_rotation is not None:
            root_rotation = global_rotation
            if pelvis_rotation is not None:
                root_rotation = SO3.multiply(
                    SO3.convert(global_rotation, src="axis_angle", dst="quat", xp=np),
                    SO3.convert(pelvis_rotation, src="axis_angle", dst="quat", xp=np),
                    xp=np,
                )
                root_rotation = SO3.convert(root_rotation, src="quat", dst="axis_angle", xp=np)
            motion["global_rotation"] = root_rotation
        return motion

    def to_smpl_motion(self, qpos: Float[np.ndarray, "B Q"]) -> dict[str, Float[np.ndarray, "..."]]:
        coord = np.asarray(self.mujoco_to_model, dtype=qpos.dtype)
        model_to_mujoco = coord.T
        root_rot_mujoco = SO3.conversions.from_quat_to_rotmat(qpos[..., 3:7], convention="wxyz", xp=np)
        root_rot = coord @ root_rot_mujoco @ model_to_mujoco
        ordered = SO3.conversions.from_euler_to_axis_angle(
            qpos[..., 7:].reshape(*qpos.shape[:-1], len(BODY_JOINTS), 3),
            convention="XYZ",
            xp=np,
        )
        smpl_body_pose = np.zeros((*qpos.shape[:-1], 23, 3), dtype=qpos.dtype)
        for joint_index, (_, smpl_index) in enumerate(BODY_JOINTS):
            smpl_body_pose[..., smpl_index, :] = ordered[..., joint_index, :]
        return {
            "smpl_body_pose": smpl_body_pose,
            "global_translation": np.squeeze(coord @ qpos[..., :3, None], axis=-1),
            "global_rotation": SO3.conversions.from_rotmat_to_axis_angle(root_rot, xp=np),
        }

    def _preset_pose(self, name: str, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        axis_angle = np.asarray(SMPL_BODY_PRESETS[name], dtype=params["body_pose"].dtype)
        ordered = np.stack([axis_angle[smpl_index] for _, smpl_index in BODY_JOINTS])
        ordered = SO3.conversions.from_axis_angle_to_euler(ordered, convention="XYZ", xp=np).reshape(-1)
        params["body_pose"] = np.broadcast_to(ordered, (*batch_dims, ordered.shape[0])).copy()
        return params
