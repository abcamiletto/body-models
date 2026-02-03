"""PyTorch backend for SMPL model."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from ..base import BodyModel
from . import core
from .io import get_model_path, load_model_data, simplify_mesh, compute_kinematic_fronts


class SMPL(BodyModel, nn.Module):
    """SMPL body model with PyTorch backend."""

    NUM_BODY_JOINTS = 23
    NUM_JOINTS = 24

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
        gender: str | None = None,
        simplify: float = 1.0,
        ground_plane: bool = True,
    ):
        if gender is not None and gender not in ("neutral", "male", "female"):
            raise ValueError(f"Invalid gender: {gender}. Must be 'neutral', 'male', or 'female'.")
        assert simplify >= 1.0
        super().__init__()

        # Default gender to "neutral" for attribute storage when model_path is given
        self.gender = gender if gender is not None else "neutral"
        self.ground_plane = ground_plane

        resolved_path = get_model_path(model_path, gender)
        data = load_model_data(resolved_path)

        v_template_full = np.asarray(data["v_template"], dtype=np.float32)
        faces = np.asarray(data["f"], dtype=np.int32)
        lbs_weights = np.asarray(data["weights"], dtype=np.float32)
        shapedirs_full = np.asarray(data["shapedirs"], dtype=np.float32)
        shapedirs = shapedirs_full
        posedirs = np.asarray(data["posedirs"], dtype=np.float32)
        J_regressor = np.asarray(data["J_regressor"], dtype=np.float32)
        parents = np.asarray(data["kintree_table"][0], dtype=np.int32)

        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            v_template, faces, vertex_map = simplify_mesh(v_template_full, faces, target_faces)
            lbs_weights = lbs_weights[vertex_map]
            shapedirs = shapedirs_full[vertex_map]
            posedirs = posedirs[vertex_map]
        else:
            v_template = v_template_full

        # Register buffers for device management
        self.register_buffer("v_template", torch.as_tensor(v_template))
        self.register_buffer("v_template_full", torch.as_tensor(v_template_full))
        self.register_buffer("lbs_weights", torch.as_tensor(lbs_weights))
        self.register_buffer("J_regressor", torch.as_tensor(J_regressor))
        self.register_buffer("parents", torch.as_tensor(parents))
        self.register_buffer("_faces", torch.as_tensor(faces))

        # Use nn.Parameter for blend shapes (for proper device handling)
        self.shapedirs = nn.Parameter(torch.as_tensor(shapedirs), requires_grad=False)
        self.shapedirs_full = nn.Parameter(torch.as_tensor(shapedirs_full), requires_grad=False)
        self.posedirs = nn.Parameter(torch.as_tensor(posedirs.reshape(-1, posedirs.shape[-1]).T), requires_grad=False)

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
    def skin_weights(self) -> Float[Tensor, "V 24"]:
        return self.lbs_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.v_template

    def forward_vertices(
        self,
        shape: Float[Tensor, "B|1 10"],
        body_pose: Float[Tensor, "B 23 3"],
        pelvis_rotation: Float[Tensor, "B 3"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B V 3"]:
        return core.forward_vertices(
            v_template=self.v_template,
            v_template_full=self.v_template_full,
            shapedirs=self.shapedirs,
            shapedirs_full=self.shapedirs_full,
            posedirs=self.posedirs,
            lbs_weights=self.lbs_weights,
            J_regressor=self.J_regressor,
            parents=self.parents,
            kinematic_fronts=self._kinematic_fronts,
            shape=shape,
            body_pose=body_pose,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            ground_plane=self.ground_plane,
        )

    def forward_skeleton(
        self,
        shape: Float[Tensor, "B|1 10"],
        body_pose: Float[Tensor, "B 23 3"],
        pelvis_rotation: Float[Tensor, "B 3"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B 24 4 4"]:
        return core.forward_skeleton(
            v_template_full=self.v_template_full,
            shapedirs_full=self.shapedirs_full,
            J_regressor=self.J_regressor,
            parents=self.parents,
            kinematic_fronts=self._kinematic_fronts,
            shape=shape,
            body_pose=body_pose,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            ground_plane=self.ground_plane,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        device = self.v_template.device
        return {
            "shape": torch.zeros((1, 10), device=device, dtype=dtype),
            "body_pose": torch.zeros((batch_size, self.NUM_BODY_JOINTS, 3), device=device, dtype=dtype),
            "pelvis_rotation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
            "global_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }


# Re-export conversion functions from core
from_native_args = core.from_native_args
to_native_outputs = core.to_native_outputs
