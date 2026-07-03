"""PyTorch backend for the SMPL humanoid robot."""

from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor
from trimesh import Trimesh

from body_models import common
from body_models.base import RigidBodyModel
from body_models.robots.smpl_humanoid.backends import torch as backend
from body_models.robots.smpl_humanoid.constants import BODY_JOINTS, SMPL_BODY_PRESETS, SMPL_HUMANOID_JOINTS
from body_models.robots.smpl_humanoid.io import load_model_data

__all__ = ["SmplHumanoid"]


class SmplHumanoid(RigidBodyModel, nn.Module):
    """Rigid humanoid loaded from MJCF using the canonical 24-joint SMPL hierarchy."""

    JOINTS = SMPL_HUMANOID_JOINTS

    def __init__(
        self,
        source: Path | str = "humenv",
    ) -> None:
        super().__init__()
        self.weights = common.torchify(load_model_data(source))

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
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
    def actuated_joint_limits(self) -> Float[Tensor, "Q 2"]:
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
    def link_geom_positions(self) -> Float[Tensor, "L 3"]:
        return self.weights.link_geom_positions

    @property
    def link_geom_rotations(self) -> Float[Tensor, "L 3 3"]:
        return self.weights.link_geom_rotations

    @property
    def num_vertices(self) -> int:
        return self.weights.vertices.shape[0]

    def forward_skeleton(
        self,
        body_pose: Float[Tensor, "B Q"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[Tensor, "B 24 4 4"]:
        return backend.forward_skeleton(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
        )

    def forward_meshes(
        self,
        body_pose: Float[Tensor, "B Q"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B 3"] | None = None,
    ) -> list[Trimesh]:
        return backend.forward_meshes(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
        )

    def forward_links(
        self,
        body_pose: Float[Tensor, "B Q"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B 24 4 4"]:
        return backend.forward_links(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        device = self.weights.vertices.device
        return {
            "body_pose": torch.zeros((*batch_dims, self.num_actuated), device=device, dtype=dtype),
            "global_rotation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
            "global_translation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
        }

    def get_tpose(self, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, Tensor]:
        return self._preset_pose("t_pose", batch_dims=batch_dims, **kwargs)

    def get_apose(self, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, Tensor]:
        return self._preset_pose("a_pose", batch_dims=batch_dims, **kwargs)

    def from_smpl_motion(
        self,
        smpl_body_pose: Float[Tensor, "B 23 3"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        pelvis_rotation: Float[Tensor, "B 3"] | None = None,
    ) -> dict[str, Tensor]:
        ordered = torch.stack([smpl_body_pose[..., smpl_index, :] for _, smpl_index in BODY_JOINTS], dim=-2)
        motion = {
            "body_pose": SO3.conversions.from_axis_angle_to_euler(ordered, convention="XYZ", xp=torch).reshape(
                *smpl_body_pose.shape[:-2], self.num_actuated
            )
        }
        if global_translation is not None:
            motion["global_translation"] = global_translation
        if global_rotation is not None:
            root_rotation = global_rotation
            if pelvis_rotation is not None:
                root_rotation = SO3.multiply(
                    SO3.convert(global_rotation, src="axis_angle", dst="quat", xp=torch),
                    SO3.convert(pelvis_rotation, src="axis_angle", dst="quat", xp=torch),
                    xp=torch,
                )
                root_rotation = SO3.convert(root_rotation, src="quat", dst="axis_angle", xp=torch)
            motion["global_rotation"] = root_rotation
        return motion

    def to_smpl_motion(self, qpos: Float[Tensor, "B Q"]) -> dict[str, Float[Tensor, "..."]]:
        coord = torch.as_tensor(self.mujoco_to_model, device=qpos.device, dtype=qpos.dtype)
        model_to_mujoco = coord.T
        root_rot_mujoco = SO3.conversions.from_quat_to_rotmat(qpos[..., 3:7], convention="wxyz", xp=torch)
        root_rot = coord @ root_rot_mujoco @ model_to_mujoco
        ordered = SO3.conversions.from_euler_to_axis_angle(
            qpos[..., 7:].reshape(*qpos.shape[:-1], len(BODY_JOINTS), 3),
            convention="XYZ",
            xp=torch,
        )
        smpl_body_pose = torch.zeros((*qpos.shape[:-1], 23, 3), device=qpos.device, dtype=qpos.dtype)
        for joint_index, (_, smpl_index) in enumerate(BODY_JOINTS):
            smpl_body_pose[..., smpl_index, :] = ordered[..., joint_index, :]
        return {
            "smpl_body_pose": smpl_body_pose,
            "global_translation": torch.squeeze(coord @ qpos[..., :3, None], dim=-1),
            "global_rotation": SO3.conversions.from_rotmat_to_axis_angle(root_rot, xp=torch),
        }

    def _preset_pose(self, name: str, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, Tensor]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        axis_angle = torch.as_tensor(
            SMPL_BODY_PRESETS[name],
            device=self.weights.vertices.device,
            dtype=params["body_pose"].dtype,
        )
        ordered = torch.stack([axis_angle[smpl_index] for _, smpl_index in BODY_JOINTS])
        ordered = SO3.conversions.from_axis_angle_to_euler(ordered, convention="XYZ", xp=torch).reshape(-1)
        params["body_pose"] = torch.broadcast_to(ordered, (*batch_dims, ordered.shape[0]))
        return params
