"""PyTorch backend for FLAME model."""

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from body_models import common
from body_models.base import BodyModel
from nanomanifold import SO3

from body_models.flame.backends import torch as torch_backend
from body_models.flame.constants import FLAME_JOINT_NAMES
from body_models.flame.io import get_model_path, load_model_data
from body_models.rotations import VALID_ROTATION_TYPES, RotationType

__all__ = ["FLAME"]


class FLAME(BodyModel, nn.Module):
    """FLAME head model with PyTorch backend."""

    NUM_HEAD_JOINTS = 4
    NUM_JOINTS = 5
    kernels = ("torch", "warp")

    def __init__(
        self,
        model_path: Path | str | None = None,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        kernel: Literal["torch", "warp"] = "torch",
    ):
        """Initialize the FLAME model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            simplify: Mesh simplification factor to apply while loading.
            rotation_type: Rotation representation expected by pose inputs.
            kernel: Backend kernel used for forward evaluation.
        """
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if kernel not in self.kernels:
            raise ValueError(f"Invalid kernel: {kernel}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")
        super().__init__()
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self._kernel = _get_kernel(kernel)

        resolved_path = get_model_path(model_path)
        weights = load_model_data(resolved_path, simplify=simplify)
        self.weights = common.torchify(weights)

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return list(FLAME_JOINT_NAMES)

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[Tensor, "V 5"]:
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
    def lbs_weights(self) -> Float[Tensor, "V 5"]:
        return self.weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        shape: Float[Tensor, "B|1 S"],
        expression: Float[Tensor, "B E"],
        head_pose: Float[Tensor, "B 4 N"] | Float[Tensor, "B 4 3 3"],
        head_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[Tensor, "B V 3"]:
        """Compute posed mesh vertices.

        Args:
            shape: Shape coefficients.
            expression: Facial expression coefficients.
            head_pose: Local head and facial joint rotations.
            head_rotation: Root head rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            vertex_indices: Optional subset of vertices to return.

        Returns:
            Posed vertex positions.
        """
        return self._kernel.forward_vertices(
            weights=self.weights,
            shape=shape,
            expression=expression,
            pose=head_pose,
            head_rotation=head_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[Tensor, "B|1 S"],
        expression: Float[Tensor, "B E"],
        head_pose: Float[Tensor, "B 4 N"] | Float[Tensor, "B 4 3 3"],
        head_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[Tensor, "B 5 4 4"]:
        """Compute posed joint transforms.

        Args:
            shape: Shape coefficients.
            expression: Facial expression coefficients.
            head_pose: Local head and facial joint rotations.
            head_rotation: Root head rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
        return self._kernel.forward_skeleton(
            weights=self.weights,
            shape=shape,
            expression=expression,
            pose=head_pose,
            head_rotation=head_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype=torch.float32) -> dict[str, Tensor]:
        device = self.rest_vertices.device
        ref = torch.zeros((*batch_dims, 100), device=device, dtype=dtype)
        return {
            "shape": torch.zeros((*batch_dims, 300), device=device, dtype=dtype),
            "expression": torch.zeros((*batch_dims, 100), device=device, dtype=dtype),
            "head_pose": SO3.identity_as(
                ref,
                batch_dims=(*batch_dims, self.NUM_HEAD_JOINTS),
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "head_rotation": SO3.identity_as(ref, batch_dims=batch_dims, rotation_type=self.rotation_type, xp=torch),
            "global_rotation": SO3.identity_as(ref, batch_dims=batch_dims, rotation_type=self.rotation_type, xp=torch),
            "global_translation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
        }


def _get_kernel(kernel: Literal["torch", "warp"]):
    if kernel == "torch":
        return torch_backend

    try:
        from body_models.flame.backends import warp as warp_backend
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install body-models[warp] to use FLAME kernel='warp'.") from exc

    return warp_backend
