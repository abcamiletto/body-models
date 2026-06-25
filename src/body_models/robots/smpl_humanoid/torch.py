"""PyTorch backend for the procedural SMPL humanoid robot."""

from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from body_models import common
from body_models.base import BodyModel
from body_models.robots.smpl_humanoid.backends import core
from body_models.robots.smpl_humanoid.backends import torch as backend
from body_models.robots.smpl_humanoid.constants import BODY_JOINTS, SMPL_BODY_PRESETS, SMPL_HUMANOID_JOINTS
from body_models.robots.smpl_humanoid.io import ACTION_SIZE, QPOS_SIZE, load_model_data

__all__ = ["SmplHumanoid"]
BODY_POSE_ORDER = [smpl_index for _, smpl_index in BODY_JOINTS]


class SmplHumanoid(BodyModel, nn.Module):
    """Rigid procedural humanoid using the canonical 24-joint SMPL hierarchy."""

    is_rigid_body = True
    JOINTS = SMPL_HUMANOID_JOINTS

    def __init__(
        self, model_path: Path | str | None = None, *, rotation_type: core.RotationType = "axis_angle"
    ) -> None:
        if rotation_type not in core.VALID_ROTATION_TYPES or rotation_type == "hinge":
            raise ValueError(f"Invalid rotation_type for SmplHumanoid: {rotation_type}")
        super().__init__()
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self.weights = common.torchify(load_model_data(model_path))
        self.register_buffer("_pd_action_offset", torch.zeros(ACTION_SIZE, dtype=torch.float32))
        self.register_buffer("_pd_action_scale", torch.full((ACTION_SIZE,), torch.pi, dtype=torch.float32))

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
    def qpos_joint_names(self) -> list[str]:
        return self.weights.qpos_joint_names

    @property
    def qpos_joint_indices(self) -> list[int]:
        return self.weights.qpos_joint_indices

    @property
    def qpos_joint_limits(self) -> Float[Tensor, "Q 2"]:
        return self.weights.qpos_joint_limits

    @property
    def link_names(self) -> list[str]:
        return self.weights.link_names

    @property
    def num_vertices(self) -> int:
        return self.weights.vertices.shape[0]

    @property
    def skin_weights(self) -> Float[Tensor, "V J"]:
        raise NotImplementedError(core.SKIN_WEIGHTS_ERROR)

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        params = self.get_rest_pose(batch_dims=())
        return self.forward_vertices(
            body_pose=params["body_pose"],
            global_translation=params["global_translation"],
            global_rotation=params["global_rotation"],
        )

    @property
    def pd_action_offset(self) -> Tensor:
        return cast(Tensor, self._pd_action_offset)

    @property
    def pd_action_scale(self) -> Tensor:
        return cast(Tensor, self._pd_action_scale)

    def forward_skeleton(
        self,
        body_pose: Float[Tensor, "B 23 N"] | Float[Tensor, "B 23 3 3"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[Tensor, "B 24 4 4"]:
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
        body_pose: Float[Tensor, "B 23 N"] | Float[Tensor, "B 23 3 3"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        vertex_indices: list[int] | None = None,
    ) -> Float[Tensor, "B V 3"]:
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
        body_pose: Float[Tensor, "B 23 N"] | Float[Tensor, "B 23 3 3"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    ) -> Float[Tensor, "B 24 4 4"]:
        return backend.forward_links(
            self.weights,
            self._qpos_order_body_pose(body_pose),
            global_translation,
            global_rotation=global_rotation,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        device = self.weights.vertices.device
        pose_ref = torch.zeros((*batch_dims, len(BODY_JOINTS), 3), device=device, dtype=dtype)
        global_ref = torch.zeros((*batch_dims, 3), device=device, dtype=dtype)
        return {
            "body_pose": SO3.identity_as(
                pose_ref,
                batch_dims=(*batch_dims, len(BODY_JOINTS)),
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "global_rotation": SO3.identity_as(
                global_ref,
                batch_dims=batch_dims,
                rotation_type=core.GLOBAL_ROTATION_TYPES[self.rotation_type],
                xp=torch,
            ),
            "global_translation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
        }

    def get_tpose(self, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, Tensor]:
        return self._preset_pose("t_pose", batch_dims=batch_dims, **kwargs)

    def get_apose(self, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, Tensor]:
        return self._preset_pose("a_pose", batch_dims=batch_dims, **kwargs)

    def _preset_pose(self, name: str, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, Tensor]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        axis_angle = torch.as_tensor(
            SMPL_BODY_PRESETS[name],
            device=self.weights.vertices.device,
            dtype=params["body_pose"].dtype,
        )
        axis_angle = torch.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=torch)
        return params

    def action_to_pd_target(self, action: Float[Tensor, "... 69"]) -> Float[Tensor, "... 69"]:
        return self.pd_action_offset.to(action) + self.pd_action_scale.to(action) * torch.tanh(action)

    def joint_pos_to_action(self, joint_pos: Float[Tensor, "... 69"]) -> Float[Tensor, "... 69"]:
        scaled = torch.clip(
            (joint_pos - self.pd_action_offset.to(joint_pos)) / self.pd_action_scale.to(joint_pos),
            -0.999,
            0.999,
        )
        return torch.atanh(scaled)

    def to_qpos(
        self,
        body_pose: Float[Tensor, "... 23 3"],
        global_translation: Float[Tensor, "... 3"] | None = None,
        global_rotation: Float[Tensor, "... 3"] | None = None,
    ) -> Float[Tensor, "... 76"]:
        if self.rotation_type != "axis_angle":
            raise ValueError("to_qpos expects an axis_angle SmplHumanoid.")
        batch_shape = body_pose.shape[:-2]
        if global_translation is None:
            global_translation = torch.zeros((*batch_shape, 3), device=body_pose.device, dtype=body_pose.dtype)
        if global_rotation is None:
            global_rotation = torch.zeros((*batch_shape, 3), device=body_pose.device, dtype=body_pose.dtype)
        root_quat = SO3.convert(global_rotation, src="axis_angle", dst="quat", xp=torch)
        flat_body_pose = body_pose.reshape(-1, *body_pose.shape[-2:])
        ordered = [flat_body_pose[:, smpl_index] for _, smpl_index in BODY_JOINTS]
        joint_pos = torch.stack(ordered, dim=1).reshape(*batch_shape, ACTION_SIZE)
        return torch.concat([global_translation, root_quat, joint_pos], dim=-1).reshape(*batch_shape, QPOS_SIZE)

    def _qpos_order_body_pose(self, body_pose: Tensor) -> Tensor:
        indices = torch.as_tensor(BODY_POSE_ORDER, device=body_pose.device, dtype=torch.long)
        dim = -3 if self.rotation_type in ("matrix", "rotmat") else -2
        return torch.index_select(body_pose, dim, indices)
