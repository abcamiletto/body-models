"""NumPy backend for the procedural SMPL humanoid robot."""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models.base import BodyModel
from body_models.robots.smpl_humanoid.backends import core
from body_models.robots.smpl_humanoid.backends import numpy as backend
from body_models.robots.smpl_humanoid.constants import BODY_JOINTS, SMPL_BODY_PRESETS, SMPL_HUMANOID_JOINTS
from body_models.robots.smpl_humanoid.io import ACTION_SIZE, QPOS_SIZE, load_model_data

__all__ = ["SmplHumanoid"]
BODY_POSE_ORDER = [smpl_index for _, smpl_index in BODY_JOINTS]


class SmplHumanoid(BodyModel):
    """Rigid procedural humanoid using the canonical 24-joint SMPL hierarchy."""

    is_rigid_body = True
    JOINTS = SMPL_HUMANOID_JOINTS

    def __init__(
        self, model_path: Path | str | None = None, *, rotation_type: core.RotationType = "axis_angle"
    ) -> None:
        if rotation_type not in core.VALID_ROTATION_TYPES or rotation_type == "hinge":
            raise ValueError(f"Invalid rotation_type for SmplHumanoid: {rotation_type}")
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self.weights = load_model_data(model_path)
        self.pd_action_offset = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.pd_action_scale = np.full(ACTION_SIZE, np.pi, dtype=np.float32)

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
    def qpos_joint_names(self) -> list[str]:
        return self.weights.qpos_joint_names

    @property
    def qpos_joint_indices(self) -> list[int]:
        return self.weights.qpos_joint_indices

    @property
    def qpos_joint_limits(self) -> Float[np.ndarray, "Q 2"]:
        return self.weights.qpos_joint_limits

    @property
    def link_names(self) -> list[str]:
        return self.weights.link_names

    @property
    def num_vertices(self) -> int:
        return self.weights.vertices.shape[0]

    @property
    def skin_weights(self) -> Float[np.ndarray, "V J"]:
        raise NotImplementedError(core.SKIN_WEIGHTS_ERROR)

    @property
    def rest_vertices(self) -> Float[np.ndarray, "V 3"]:
        params = self.get_rest_pose(batch_dims=())
        return self.forward_vertices(
            body_pose=params["body_pose"],
            global_translation=params["global_translation"],
            global_rotation=params["global_rotation"],
        )

    def forward_skeleton(
        self,
        body_pose: Float[np.ndarray, "B 23 N"] | Float[np.ndarray, "B 23 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[np.ndarray, "B 24 4 4"]:
        return backend.forward_skeleton(
            self.weights,
            self._qpos_order_body_pose(body_pose),
            global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def forward_vertices(
        self,
        body_pose: Float[np.ndarray, "B 23 N"] | Float[np.ndarray, "B 23 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        vertex_indices: list[int] | None = None,
    ) -> Float[np.ndarray, "B V 3"]:
        return backend.forward_vertices(
            self.weights,
            self._qpos_order_body_pose(body_pose),
            global_translation,
            global_rotation=global_rotation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_links(
        self,
        body_pose: Float[np.ndarray, "B 23 N"] | Float[np.ndarray, "B 23 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    ) -> Float[np.ndarray, "B 24 4 4"]:
        return backend.forward_links(
            self.weights,
            self._qpos_order_body_pose(body_pose),
            global_translation,
            global_rotation=global_rotation,
            rotation_type=self.rotation_type,
        )

    def link_mesh(self, link_name: str) -> dict[str, np.ndarray | str | int]:
        return core.link_mesh(
            vertices=self.weights.vertices,
            faces=self.weights.faces,
            link_joint_indices=self.weights.link_joint_indices,
            link_vertex_starts=self.weights.link_vertex_starts,
            link_vertex_counts=self.weights.link_vertex_counts,
            link_face_starts=self.weights.link_face_starts,
            link_face_counts=self.weights.link_face_counts,
            joint_names=self.weights.joint_names,
            link_names=self.weights.link_names,
            link_name=link_name,
        )

    def joint_meshes(self, joint_name: str) -> list[dict[str, np.ndarray | str | int]]:
        return core.joint_meshes(
            vertices=self.weights.vertices,
            faces=self.weights.faces,
            link_joint_indices=self.weights.link_joint_indices,
            link_vertex_starts=self.weights.link_vertex_starts,
            link_vertex_counts=self.weights.link_vertex_counts,
            link_face_starts=self.weights.link_face_starts,
            link_face_counts=self.weights.link_face_counts,
            joint_names=self.weights.joint_names,
            link_names=self.weights.link_names,
            joint_name=joint_name,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype=np.float32) -> dict[str, np.ndarray]:
        pose_ref = np.zeros((*batch_dims, len(BODY_JOINTS), 3), dtype=dtype)
        body_pose = SO3.identity_as(
            pose_ref,
            batch_dims=(*batch_dims, len(BODY_JOINTS)),
            rotation_type=self.rotation_type,
            xp=np,
        ).copy()
        global_ref = np.zeros((*batch_dims, 3), dtype=dtype)
        return {
            "body_pose": body_pose,
            "global_rotation": SO3.identity_as(
                global_ref,
                batch_dims=batch_dims,
                rotation_type=core.GLOBAL_ROTATION_TYPES[self.rotation_type],
                xp=np,
            ).copy(),
            "global_translation": np.zeros((*batch_dims, 3), dtype=dtype),
        }

    def get_tpose(self, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, np.ndarray]:
        return self._preset_pose("t_pose", batch_dims=batch_dims, **kwargs)

    def get_apose(self, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, np.ndarray]:
        return self._preset_pose("a_pose", batch_dims=batch_dims, **kwargs)

    def _preset_pose(self, name: str, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        axis_angle = np.asarray(SMPL_BODY_PRESETS[name], dtype=params["body_pose"].dtype)
        axis_angle = np.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=np).copy()
        return params

    def action_to_pd_target(self, action: Float[np.ndarray, "... 69"]) -> Float[np.ndarray, "... 69"]:
        return self.pd_action_offset + self.pd_action_scale * np.tanh(action)

    def joint_pos_to_action(self, joint_pos: Float[np.ndarray, "... 69"]) -> Float[np.ndarray, "... 69"]:
        scaled = np.clip((joint_pos - self.pd_action_offset) / self.pd_action_scale, -0.999, 0.999)
        return np.arctanh(scaled)

    def to_qpos(
        self,
        body_pose: Float[np.ndarray, "... 23 3"],
        global_translation: Float[np.ndarray, "... 3"] | None = None,
        global_rotation: Float[np.ndarray, "... 3"] | None = None,
    ) -> Float[np.ndarray, "... 76"]:
        if self.rotation_type != "axis_angle":
            raise ValueError("to_qpos expects an axis_angle SmplHumanoid.")
        batch_shape = body_pose.shape[:-2]
        if global_translation is None:
            global_translation = np.zeros((*batch_shape, 3), dtype=body_pose.dtype)
        if global_rotation is None:
            global_rotation = np.zeros((*batch_shape, 3), dtype=body_pose.dtype)
        root_quat = SO3.convert(global_rotation, src="axis_angle", dst="quat", xp=np)
        flat_body_pose = body_pose.reshape(-1, *body_pose.shape[-2:])
        ordered = [flat_body_pose[:, smpl_index] for _, smpl_index in BODY_JOINTS]
        joint_pos = np.stack(ordered, axis=1).reshape(*batch_shape, ACTION_SIZE)
        return np.concatenate([global_translation, root_quat, joint_pos], axis=-1).reshape(*batch_shape, QPOS_SIZE)

    def _qpos_order_body_pose(self, body_pose: np.ndarray) -> np.ndarray:
        if self.rotation_type in ("matrix", "rotmat"):
            return body_pose[..., BODY_POSE_ORDER, :, :]
        return body_pose[..., BODY_POSE_ORDER, :]
