"""PyTorch backend for MHR model."""

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from body_models import common
from body_models.base import SkinnedModel
from body_models.bodies.mhr.backends import torch as backend
from body_models.bodies.mhr.backends.core import MhrIdentity, MhrPreparedPose
from body_models.bodies.mhr.constants import (
    MHR_BODY_POSE_DIM,
    MHR_HEAD_POSE_DIM,
    MHR_HAND_PRESETS,
    MHR_HAND_POSE_DIM,
    MHR_BODY_PRESETS,
    MHR_JOINTS,
)
from body_models.bodies.mhr.io import get_model_path, load_model_data
from body_models.bodies.mhr.pose import pack_pose, unpack_pose

__all__ = ["MHR"]


class MHR(SkinnedModel, nn.Module):
    """MHR body model with PyTorch backend."""

    has_hands = True
    has_head = True
    SHAPE_DIM = 45
    EXPR_DIM = 72
    JOINTS = MHR_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        lod: int = 1,
        simplify: float = 1.0,
    ) -> None:
        """Initialize the MHR model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            lod: Level-of-detail variant to load.
            simplify: Mesh simplification factor to apply while loading.
        """
        super().__init__()
        self.weights = common.torchify(load_model_data(get_model_path(model_path), lod=lod, simplify=simplify))

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.parents)

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self.weights.base_vertices.shape[0]

    @property
    def pose_dim(self) -> int:
        return self.weights.parameter_transform.shape[1] - self.SHAPE_DIM

    @property
    def body_pose_dim(self) -> int:
        return MHR_BODY_POSE_DIM

    @property
    def head_pose_dim(self) -> int:
        return MHR_HEAD_POSE_DIM

    @property
    def hand_pose_dim(self) -> int:
        return MHR_HAND_POSE_DIM

    @property
    def skin_weights(self) -> Float[Tensor, "V J"]:
        dense = torch.zeros(
            self.weights.skin_weights.shape[0],
            self.num_joints,
            device=self.weights.skin_weights.device,
            dtype=self.weights.skin_weights.dtype,
        )
        dense.scatter_(1, self.weights.skin_indices, self.weights.skin_weights)
        return dense

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.weights.base_vertices * 0.01

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        body_pose: Float[Tensor, "*batch 94"],
        head_pose: Float[Tensor, "*batch 6"],
        hand_pose: Float[Tensor, "*batch 104"],
        expression: Float[Tensor, "*batch 72"],
        global_rotation: Float[Tensor, "*batch 3"] | None = None,
        global_translation: Float[Tensor, "*batch 3"] | None = None,
        vertex_indices: Any | None = None,
        *,
        shape: Float[Tensor, "*batch 45"] | None = None,
        identity: MhrIdentity | None = None,
    ) -> Float[Tensor, "*batch V 3"]:
        """Compute posed mesh vertices.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            head_pose: Local head and facial controls.
            hand_pose: Local hand joint rotations.
            expression: Facial expression coefficients.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            vertex_indices: Optional subset of vertices to return.

        Returns:
            Posed vertex positions.
        """
        if identity is None:
            assert shape is not None
            batch_shape = body_pose.shape[:-1]
            shape = torch.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            expression = torch.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
            identity = self.prepare_identity(shape, expression=expression)
        pose = self.prepare_pose(body_pose, head_pose, hand_pose)
        return backend.forward_vertices(
            weights=self.weights,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rest_vertices=identity["rest_vertices"],
            skinning_transforms=pose["skinning_transforms"],
            pose_offsets=pose["pose_offsets"],
        )

    def forward_skeleton(
        self,
        body_pose: Float[Tensor, "*batch 94"],
        head_pose: Float[Tensor, "*batch 6"],
        hand_pose: Float[Tensor, "*batch 104"],
        expression: Float[Tensor, "*batch 72"],
        global_rotation: Float[Tensor, "*batch 3"] | None = None,
        global_translation: Float[Tensor, "*batch 3"] | None = None,
        joint_indices: Any | None = None,
        *,
        shape: Float[Tensor, "*batch 45"] | None = None,
        identity: MhrIdentity | None = None,
    ) -> Float[Tensor, "*batch J 4 4"]:
        """Compute posed joint transforms.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            head_pose: Local head and facial controls.
            hand_pose: Local hand joint rotations.
            expression: Facial expression coefficients.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
        if identity is None:
            assert shape is not None
            batch_shape = body_pose.shape[:-1]
            shape = torch.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            expression = torch.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
            identity = self.prepare_identity(shape, expression=expression, skip_vertices=True)
        pose = self.prepare_pose(body_pose, head_pose, hand_pose, skip_vertices=True)
        return backend.forward_skeleton(
            weights=self.weights,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            skeleton_transforms=pose["skeleton_transforms"],
        )

    def prepare_identity(
        self,
        shape: Float[Tensor, "*batch 45"],
        expression: Float[Tensor, "*batch 72"],
        skip_vertices: bool = False,
    ) -> MhrIdentity:
        """Precompute shape- and expression-dependent state for repeated forward passes."""
        return backend.prepare_identity(self.weights, shape, expression=expression, skip_vertices=skip_vertices)

    def prepare_pose(
        self,
        body_pose: Float[Tensor, "*batch 94"],
        head_pose: Float[Tensor, "*batch 6"],
        hand_pose: Float[Tensor, "*batch 104"],
        *,
        identity: MhrIdentity | None = None,
        skip_vertices: bool = False,
    ) -> MhrPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        pose = pack_pose(torch, body_pose, head_pose, hand_pose)
        return backend.prepare_pose(self.weights, pose, skip_vertices=skip_vertices)

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Tensor]:
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        device = self.rest_vertices.device
        hand_pose = torch.zeros((*batch_dims, self.hand_pose_dim), device=device, dtype=dtype)
        if hands != "default":
            preset = MHR_HAND_PRESETS[hands]
            hand_pose = torch.asarray(preset, device=device, dtype=dtype).reshape(self.hand_pose_dim)
            hand_pose = torch.broadcast_to(hand_pose, (*batch_dims, self.hand_pose_dim))
        return {
            "shape": torch.zeros((*batch_dims, self.SHAPE_DIM), device=device, dtype=dtype),
            "body_pose": torch.zeros((*batch_dims, self.body_pose_dim), device=device, dtype=dtype),
            "head_pose": torch.zeros((*batch_dims, self.head_pose_dim), device=device, dtype=dtype),
            "hand_pose": hand_pose,
            "expression": torch.zeros((*batch_dims, self.EXPR_DIM), device=device, dtype=dtype),
            "global_rotation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
            "global_translation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
        }

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, Tensor]:
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        pose = torch.zeros(
            (*batch_dims, self.pose_dim), device=params["body_pose"].device, dtype=params["body_pose"].dtype
        )
        pose[..., :100] = torch.as_tensor(MHR_BODY_PRESETS["t_pose"], device=pose.device, dtype=pose.dtype)
        params["body_pose"], params["head_pose"], _hand_pose = unpack_pose(torch, pose)
        return params

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, Tensor]:
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
