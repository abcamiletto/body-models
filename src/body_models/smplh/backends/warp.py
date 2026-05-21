"""Warp-accelerated SMPL-H Torch backend."""

import torch
from jaxtyping import Float
from torch import Tensor

from body_models.rotations import RotationType
from body_models.smpl.backends import core as smpl_core
from body_models.smpl.backends import warp as smpl_warp
from body_models.smplh.backends import torch as torch_backend
from body_models.smplh.io import SmplhWeights

forward_skeleton = torch_backend.forward_skeleton
prepare_identity = torch_backend.prepare_identity
prepare_pose = torch_backend.prepare_pose

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def forward_vertices(
    weights: SmplhWeights,
    global_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
    global_translation: Float[Tensor, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[Tensor, "*batch J 3"],
    rest_vertices: Float[Tensor, "*batch V 3"],
    joint_transforms: Float[Tensor, "*batch J 4 4"],
    pose_offsets: Float[Tensor, "*batch V 3"],
):
    joint_indices = weights.lbs_joint_indices
    joint_weights = weights.lbs_joint_weights
    v_shaped = rest_vertices + pose_offsets
    if vertex_indices is not None:
        selected_vertices = torch.as_tensor(vertex_indices, device=rest_vertices.device)
        v_shaped = v_shaped[..., selected_vertices, :]
        joint_indices = joint_indices[selected_vertices]
        joint_weights = joint_weights[selected_vertices]

    v_posed = smpl_warp.warp_linear_blend_skinning(
        v_shaped, rest_joints, joint_transforms, joint_indices, joint_weights
    )
    return smpl_core.apply_global_transform(torch, v_posed, global_rotation, global_translation, rotation_type)
