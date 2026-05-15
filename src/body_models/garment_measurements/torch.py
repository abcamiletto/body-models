"""PyTorch backend for the GarmentMeasurements PCA body model."""

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from .. import common
from ..base import BodyModel
from ..rotations import VALID_ROTATION_TYPES, RotationType
from .backends import torch as backend
from .io import get_model_path, load_model_data
from .constants import GARMENT_BODY_PRESETS, GARMENT_HAND_PRESETS, GARMENT_JOINTS
from .pose import pack_pose, unpack_pose


__all__ = ["GarmentMeasurements"]


class GarmentMeasurements(BodyModel, nn.Module):
    """GarmentMeasurements PCA body model with FBX-derived skeleton/skinning."""

    has_hands = True

    JOINTS = GARMENT_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        super().__init__()

        self.weights = common.torchify(load_model_data(get_model_path(model_path), dtype="float32"))
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.joint_names)

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self.weights.mean_vertices.shape[0]

    @property
    def num_shape_components(self) -> int:
        return self.weights.eigenvalues.shape[0]

    @property
    def skin_weights(self) -> Float[Tensor, "V J"]:
        return self.weights.skin_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.weights.mean_vertices

    @property
    def parents(self) -> list[int]:
        return [int(parent) for parent in self.weights.parents.tolist()]

    def forward_vertices(
        self,
        shape: Float[Tensor, "B C"],
        body_pose: Float[Tensor, "B 25 N"] | Float[Tensor, "B 25 3 3"],
        head_pose: Float[Tensor, "B 3 N"] | Float[Tensor, "B 3 3 3"],
        hand_pose: Float[Tensor, "B 30 N"] | Float[Tensor, "B 30 3 3"],
        pelvis_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"],
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        vertex_indices: list[int] | None = None,
    ) -> Float[Tensor, "B V 3"]:
        pose = pack_pose(torch, pelvis_rotation, body_pose, head_pose, hand_pose)
        return backend.forward_vertices(
            weights=self.weights,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[Tensor, "B C"],
        body_pose: Float[Tensor, "B 25 N"] | Float[Tensor, "B 25 3 3"],
        head_pose: Float[Tensor, "B 3 N"] | Float[Tensor, "B 3 3 3"],
        hand_pose: Float[Tensor, "B 30 N"] | Float[Tensor, "B 30 3 3"],
        pelvis_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"],
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[Tensor, "B J 4 4"]:
        pose = pack_pose(torch, pelvis_rotation, body_pose, head_pose, hand_pose)
        return backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: torch.dtype | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Tensor]:
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        dtype = dtype or self.weights.mean_vertices.dtype
        device = self.weights.mean_vertices.device
        pose_ref = torch.zeros((*batch_dims, self.num_joints, 3), dtype=dtype, device=device)
        global_ref = torch.zeros((*batch_dims,), dtype=dtype, device=device)
        pose = SO3.identity_as(
            pose_ref,
            batch_dims=(*batch_dims, self.num_joints),
            rotation_type=self.rotation_type,
            xp=torch,
        )
        pelvis_rotation, body_pose, head_pose, hand_pose = unpack_pose(torch, pose)
        if hands != "default":
            preset = GARMENT_HAND_PRESETS[hands]
            axis_angle = torch.asarray(preset, device=device, dtype=dtype).reshape(-1, 3)
            axis_angle = torch.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
            hand_pose = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=torch)
        return {
            "shape": torch.zeros((*batch_dims, self.num_shape_components), dtype=dtype, device=device),
            "body_pose": body_pose,
            "head_pose": head_pose,
            "hand_pose": hand_pose,
            "pelvis_rotation": pelvis_rotation,
            "global_rotation": SO3.identity_as(
                global_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "global_translation": torch.zeros((*batch_dims, 3), dtype=dtype, device=device),
        }

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, Tensor]:
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        axis_angle = torch.as_tensor(
            GARMENT_BODY_PRESETS["t_pose"], device=params["body_pose"].device, dtype=params["body_pose"].dtype
        )
        axis_angle = torch.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=torch)
        return params

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, Tensor]:
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
