"""Warp-accelerated SMPL-H Torch backend."""

import torch
from jaxtyping import Float
from torch import Tensor

from body_models.rotations import RotationType
from body_models.bodies.smpl.backends import core as smpl_core
from body_models.bodies.smpl.backends import warp as smpl_warp
from body_models.bodies.smplh.backends import torch as torch_backend
from body_models.bodies.smplh.io import SmplhWeights

forward_skeleton = torch_backend.forward_skeleton
prepare_identity = torch_backend.prepare_identity
prepare_pose = torch_backend.prepare_pose

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def forward_vertices(
    weights: SmplhWeights,
    rest_vertices: Float[Tensor, "*batch V 3"],
    skinning_transforms: Float[Tensor, "*batch J 4 4"],
    pose_offsets: Float[Tensor, "*batch V 3"],
    global_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
    global_translation: Float[Tensor, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    joint_indices = weights.lbs_joint_indices
    joint_weights = weights.lbs_joint_weights
    if vertex_indices is not None:
        rest_vertices = rest_vertices[..., vertex_indices, :]
        pose_offsets = pose_offsets[..., vertex_indices, :]
        joint_indices = joint_indices[vertex_indices]
        joint_weights = joint_weights[vertex_indices]

    v_shaped = rest_vertices + pose_offsets
    v_posed = smpl_warp.warp_affine_blend_skinning(v_shaped, skinning_transforms, joint_indices, joint_weights)
    return smpl_core.apply_global_transform(torch, v_posed, global_rotation, global_translation, rotation_type)
