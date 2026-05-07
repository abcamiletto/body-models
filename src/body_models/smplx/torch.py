"""PyTorch backend for SMPL-X model."""

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from body_models import common
from body_models.base import BodyModel
from nanomanifold import SO3

from body_models.rotations import VALID_ROTATION_TYPES, RotationType
from body_models.smplx.backends import torch as backend
from body_models.smplx.io import get_model_path, load_model_data

__all__ = ["SMPLX"]


class SMPLX(BodyModel, nn.Module):
    """SMPL-X body model with PyTorch backend."""

    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 30
    NUM_HEAD_JOINTS = 3
    NUM_JOINTS = 55

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ):
        if gender is not None and gender not in ("neutral", "male", "female"):
            raise ValueError(f"Invalid gender: {gender}. Must be 'neutral', 'male', or 'female'.")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        assert simplify >= 1.0
        super().__init__()

        self.gender = gender if gender is not None else "neutral"
        self.rotation_type = rotation_type

        resolved_path = get_model_path(model_path, gender)
        weights = load_model_data(resolved_path, flat_hand_mean=flat_hand_mean, simplify=simplify)
        self.weights = common.torchify(weights)

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self.weights.joint_names

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[Tensor, "V 55"]:
        return self.weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.weights.v_template

    @property
    def shapedirs(self) -> Float[Tensor, "V 3 S"]:
        return self.weights.shapedirs

    @property
    def exprdirs(self) -> Float[Tensor, "V 3 E"]:
        return self.weights.exprdirs

    @property
    def posedirs(self) -> Float[Tensor, "P V*3"]:
        return self.weights.posedirs

    @property
    def lbs_weights(self) -> Float[Tensor, "V 55"]:
        return self.weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        shape: Float[Tensor, "B|1 10"],
        body_pose: Float[Tensor, "B 21 N"] | Float[Tensor, "B 21 3 3"],
        hand_pose: Float[Tensor, "B 30 N"] | Float[Tensor, "B 30 3 3"],
        head_pose: Float[Tensor, "B 3 N"] | Float[Tensor, "B 3 3 3"],
        expression: Float[Tensor, "B 10"] | None = None,
        pelvis_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[Tensor, "B V 3"]:
        return backend.forward_vertices(
            weights=self.weights,
            shape=shape,
            body_pose=body_pose,
            hand_pose=hand_pose,
            head_pose=head_pose,
            expression=expression,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[Tensor, "B|1 10"],
        body_pose: Float[Tensor, "B 21 N"] | Float[Tensor, "B 21 3 3"],
        hand_pose: Float[Tensor, "B 30 N"] | Float[Tensor, "B 30 3 3"],
        head_pose: Float[Tensor, "B 3 N"] | Float[Tensor, "B 3 3 3"],
        expression: Float[Tensor, "B 10"] | None = None,
        pelvis_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[Tensor, "B 55 4 4"]:
        return backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            body_pose=body_pose,
            hand_pose=hand_pose,
            head_pose=head_pose,
            expression=expression,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=torch.float32) -> dict[str, Tensor]:
        device = self.rest_vertices.device
        body_pose_ref = torch.zeros((batch_size, self.NUM_BODY_JOINTS, 3), device=device, dtype=dtype)
        hand_pose_ref = torch.zeros((batch_size, self.NUM_HAND_JOINTS, 3), device=device, dtype=dtype)
        head_pose_ref = torch.zeros((batch_size, self.NUM_HEAD_JOINTS, 3), device=device, dtype=dtype)
        pelvis_ref = torch.zeros((batch_size, 3), device=device, dtype=dtype)
        return {
            "shape": torch.zeros((1, 10), device=device, dtype=dtype),
            "body_pose": SO3.identity_as(
                body_pose_ref,
                batch_dims=(batch_size, self.NUM_BODY_JOINTS),
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "hand_pose": SO3.identity_as(
                hand_pose_ref,
                batch_dims=(batch_size, self.NUM_HAND_JOINTS),
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "head_pose": SO3.identity_as(
                head_pose_ref,
                batch_dims=(batch_size, self.NUM_HEAD_JOINTS),
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "expression": torch.zeros((batch_size, 10), device=device, dtype=dtype),
            "pelvis_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "global_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }
