"""PyTorch backend for FLAME model."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from ..base import BodyModel
from . import core
from .io import get_model_path, load_model_data, simplify_mesh, compute_kinematic_fronts


class FLAME(BodyModel, nn.Module):
    """FLAME head model with PyTorch backend.

    Args:
        model_path: Path to the FLAME model file or directory.
        simplify: Mesh simplification ratio. 1.0 = original mesh, 2.0 = half faces, etc.
        ground_plane: If True (default), dynamically offset mesh so chin is at Y=0
            regardless of head shape. If False, use native FLAME coordinates.

    Forward API:
        forward_vertices(shape, expression, pose, head_rotation, global_rotation, global_translation)
        forward_skeleton(shape, expression, pose, head_rotation, global_rotation, global_translation)

        shape: [B, 300] shape betas (can use fewer)
        expression: [B, 100] expression coefficients (can use fewer)
        pose: [B, 4, 3] axis-angle for neck, jaw, left_eye, right_eye
        head_rotation: [B, 3] optional root joint rotation (affects skeleton)
        global_rotation: [B, 3] optional post-hoc rotation (doesn't affect skeleton)
        global_translation: [B, 3] optional translation
    """

    NUM_HEAD_JOINTS = 4  # neck, jaw, left_eye, right_eye
    NUM_JOINTS = 5  # root + 4 head joints

    # Type declarations for registered buffers
    v_template: Tensor
    v_template_full: Tensor
    lbs_weights: Tensor
    J_regressor: Tensor
    parents: Tensor
    _faces: Tensor

    def __init__(
        self,
        model_path: Path | str | None = None,
        simplify: float = 1.0,
        ground_plane: bool = True,
    ):
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"
        super().__init__()
        self.ground_plane = ground_plane

        resolved_path = get_model_path(model_path)
        data = load_model_data(resolved_path)

        # Load full-resolution data
        v_template_full = np.asarray(data["v_template"], dtype=np.float32)
        faces = np.asarray(data["f"], dtype=np.int32)
        lbs_weights = np.asarray(data["weights"], dtype=np.float32)
        shapedirs_full = np.asarray(data["shapedirs"], dtype=np.float32)  # (V, 3, 400)
        posedirs = np.asarray(data["posedirs"], dtype=np.float32)
        J_regressor = np.asarray(data["J_regressor"], dtype=np.float32)
        parents = np.asarray(data["kintree_table"][0], dtype=np.int32)

        # Fix parent of root (may be -1 or large value in file)
        parents[0] = 0

        # Apply mesh simplification if requested
        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            v_template, faces, vertex_map = simplify_mesh(v_template_full, faces, target_faces)
            lbs_weights = lbs_weights[vertex_map]
            shapedirs = shapedirs_full[vertex_map]
            posedirs = posedirs[vertex_map]
        else:
            v_template = v_template_full
            shapedirs = shapedirs_full

        # Register buffers for device management
        self.register_buffer("v_template", torch.as_tensor(v_template))
        self.register_buffer("v_template_full", torch.as_tensor(v_template_full))
        self.register_buffer("lbs_weights", torch.as_tensor(lbs_weights))
        self.register_buffer("J_regressor", torch.as_tensor(J_regressor))
        self.register_buffer("parents", torch.as_tensor(parents))
        self.register_buffer("_faces", torch.as_tensor(faces))

        # FLAME 2023 has combined shape (300) + expression (100) in shapedirs
        # Use nn.Parameter for blend shapes (for proper device handling)
        shapedirs_full_t = torch.as_tensor(shapedirs_full, dtype=torch.float32)
        self.shapedirs_full = nn.Parameter(shapedirs_full_t[:, :, :300], requires_grad=False)
        self.exprdirs_full = nn.Parameter(shapedirs_full_t[:, :, 300:], requires_grad=False)

        shapedirs_t = torch.as_tensor(shapedirs, dtype=torch.float32)
        self.shapedirs = nn.Parameter(shapedirs_t[:, :, :300], requires_grad=False)
        self.exprdirs = nn.Parameter(shapedirs_t[:, :, 300:], requires_grad=False)

        posedirs_t = torch.as_tensor(posedirs, dtype=torch.float32)
        self.posedirs = nn.Parameter(posedirs_t.reshape(-1, posedirs_t.shape[-1]).T, requires_grad=False)

        self._kinematic_fronts = compute_kinematic_fronts(parents)

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self._faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def num_vertices(self) -> int:
        return self.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[Tensor, "V 5"]:
        return self.lbs_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.v_template

    def forward_vertices(
        self,
        shape: Float[Tensor, "B|1 N_shape"],
        expression: Float[Tensor, "B N_expr"] | None = None,
        pose: Float[Tensor, "B 4 3"] | None = None,
        head_rotation: Float[Tensor, "B 3"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B V 3"]:
        """Compute mesh vertices [B, V, 3]."""
        B = shape.shape[0] if shape.dim() > 1 and shape.shape[0] > 1 else (pose.shape[0] if pose is not None else 1)
        device, dtype = self.J_regressor.device, self.J_regressor.dtype

        if expression is None:
            expression = torch.zeros((B, 100), device=device, dtype=dtype)
        if pose is None:
            pose = torch.zeros((B, self.NUM_HEAD_JOINTS, 3), device=device, dtype=dtype)

        return core.forward_vertices(
            v_template=self.v_template,
            v_template_full=self.v_template_full,
            shapedirs=self.shapedirs,
            shapedirs_full=self.shapedirs_full,
            exprdirs=self.exprdirs,
            exprdirs_full=self.exprdirs_full,
            posedirs=self.posedirs,
            lbs_weights=self.lbs_weights,
            J_regressor=self.J_regressor,
            parents=self.parents,
            kinematic_fronts=self._kinematic_fronts,
            shape=shape,
            expression=expression,
            pose=pose,
            head_rotation=head_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            ground_plane=self.ground_plane,
        )

    def forward_skeleton(
        self,
        shape: Float[Tensor, "B|1 N_shape"],
        expression: Float[Tensor, "B N_expr"] | None = None,
        pose: Float[Tensor, "B 4 3"] | None = None,
        head_rotation: Float[Tensor, "B 3"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B 5 4 4"]:
        """Compute skeleton joint transforms [B, 5, 4, 4]."""
        B = shape.shape[0] if shape.dim() > 1 and shape.shape[0] > 1 else (pose.shape[0] if pose is not None else 1)
        device, dtype = self.J_regressor.device, self.J_regressor.dtype

        if expression is None:
            expression = torch.zeros((B, 100), device=device, dtype=dtype)
        if pose is None:
            pose = torch.zeros((B, self.NUM_HEAD_JOINTS, 3), device=device, dtype=dtype)

        return core.forward_skeleton(
            v_template_full=self.v_template_full,
            shapedirs_full=self.shapedirs_full,
            exprdirs_full=self.exprdirs_full,
            J_regressor=self.J_regressor,
            parents=self.parents,
            kinematic_fronts=self._kinematic_fronts,
            shape=shape,
            expression=expression,
            pose=pose,
            head_rotation=head_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            ground_plane=self.ground_plane,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        """Get rest pose parameters."""
        device = self.v_template.device
        return {
            "shape": torch.zeros((1, 300), device=device, dtype=dtype),
            "expression": torch.zeros((batch_size, 100), device=device, dtype=dtype),
            "pose": torch.zeros((batch_size, self.NUM_HEAD_JOINTS, 3), device=device, dtype=dtype),
            "head_rotation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
            "global_rotation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
            "global_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }


# Re-export conversion functions from core
from_native_args = core.from_native_args
to_native_outputs = core.to_native_outputs
