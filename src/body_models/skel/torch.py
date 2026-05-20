"""PyTorch backend for SKEL model."""

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from body_models import common
from body_models.base import BodyModel
from body_models.skel.backends import torch as backend
from body_models.skel.backends.core import SkelIdentity
from body_models.skel.io import get_model_path, load_model_data
from body_models.skel.constants import SKEL_BODY_PRESETS, SKEL_JOINTS

__all__ = ["SKEL"]


class SKEL(BodyModel, nn.Module):
    """SKEL body model with PyTorch backend."""

    NUM_BETAS = 10
    NUM_JOINTS = 24
    NUM_POSE_PARAMS = 46
    JOINTS = SKEL_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["male", "female"] | None = None,
        simplify: float = 1.0,
    ):
        """Initialize the SKEL model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            gender: Model gender variant to load.
            simplify: Mesh simplification factor to apply while loading.
        """
        if gender not in {"male", "female"}:
            raise ValueError(f"Invalid gender: {gender}. Must be 'male' or 'female'.")
        assert simplify >= 1.0
        super().__init__()

        self.gender = gender
        data = load_model_data(get_model_path(model_path, gender), simplify=simplify)
        self.weights = common.torchify(data)

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[Tensor, "V 24"]:
        return self.weights.skin_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.weights.v_template + self.weights.feet_offset

    @property
    def shapedirs(self) -> Float[Tensor, "V 3 B"]:
        return self.weights.shapedirs

    @property
    def posedirs(self) -> Float[Tensor, "P V*3"]:
        return self.weights.posedirs

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    @property
    def skeleton_faces(self) -> Int[Tensor, "Fs 3"]:
        return self.weights.skel_faces

    @property
    def _feet_offset(self) -> Float[Tensor, "3"]:
        return self.weights.feet_offset

    def forward_vertices(
        self,
        body_pose: Float[Tensor, "*batch 46"],
        global_rotation: Float[Tensor, "*batch 3"] | None = None,
        global_translation: Float[Tensor, "*batch 3"] | None = None,
        vertex_indices: Any | None = None,
        *,
        shape: Float[Tensor, "*batch 10"] | None = None,
        identity: SkelIdentity | None = None,
    ) -> Float[Tensor, "*batch V 3"]:
        """Compute posed mesh vertices.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
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
            identity = self.prepare_identity(shape)
        return backend.forward_vertices(
            weights=self.weights,
            pose=body_pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            **identity,
        )

    def forward_skeleton(
        self,
        body_pose: Float[Tensor, "*batch 46"],
        global_rotation: Float[Tensor, "*batch 3"] | None = None,
        global_translation: Float[Tensor, "*batch 3"] | None = None,
        joint_indices: Any | None = None,
        *,
        shape: Float[Tensor, "*batch 10"] | None = None,
        identity: SkelIdentity | None = None,
    ) -> Float[Tensor, "*batch 24 4 4"]:
        """Compute posed joint transforms.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
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
            identity = self.prepare_identity(shape, skip_vertices=True)
        return backend.forward_skeleton(
            weights=self.weights,
            pose=body_pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            **identity,
        )

    def prepare_identity(
        self,
        shape: Float[Tensor, "*batch 10"],
        skip_vertices: bool = False,
    ) -> SkelIdentity:
        """Precompute shape-dependent state for repeated forward passes."""
        return backend.prepare_identity(self.weights, shape, skip_vertices=skip_vertices)

    def forward_links(
        self,
        body_pose: Float[Tensor, "*batch 46"],
        global_translation: Float[Tensor, "*batch 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "*batch 3"] | None = None,
        shape: Float[Tensor, "*batch 10"] | None = None,
        identity: SkelIdentity | None = None,
    ) -> Float[Tensor, "*batch 24 4 4"]:
        return self.forward_skeleton(
            body_pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            shape=shape,
            identity=identity,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        device = self.weights.v_template.device
        return {
            "shape": torch.zeros((*batch_dims, self.NUM_BETAS), device=device, dtype=dtype),
            "body_pose": torch.zeros((*batch_dims, self.NUM_POSE_PARAMS), device=device, dtype=dtype),
            "global_rotation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
            "global_translation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
        }

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, Tensor]:
        return self.get_rest_pose(batch_dims=batch_dims, **kwargs)

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, Tensor]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        body_pose = torch.as_tensor(
            SKEL_BODY_PRESETS["a_pose"], device=params["body_pose"].device, dtype=params["body_pose"].dtype
        )
        params["body_pose"] = torch.broadcast_to(body_pose, (*batch_dims, *body_pose.shape))
        return params
