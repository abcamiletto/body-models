"""Warp-accelerated MANO Torch backend."""

import torch
from jaxtyping import Float
from torch import Tensor

from body_models.mano.backends import torch as torch_backend
from body_models.mano.io import ManoWeights
from body_models.rotations import RotationType
from body_models.smpl.backends import core as smpl_core
from body_models.smpl.backends import warp as smpl_warp

forward_skeleton = torch_backend.forward_skeleton
prepare_identity = torch_backend.prepare_identity
prepare_pose = torch_backend.prepare_pose

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def forward_vertices(
    weights: ManoWeights,
    rest_vertices: Float[Tensor, "*batch V 3"],
    skinning_transforms: Float[Tensor, "*batch J 4 4"],
    pose_offsets: Float[Tensor, "*batch V 3"],
    global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    v_shaped = rest_vertices + pose_offsets
    joint_indices = weights.lbs_joint_indices
    joint_weights = weights.lbs_joint_weights
    if vertex_indices is not None:
        v_shaped = v_shaped[..., vertex_indices, :]
        joint_indices = joint_indices[vertex_indices]
        joint_weights = joint_weights[vertex_indices]

    v_posed = smpl_warp.warp_affine_blend_skinning(v_shaped, skinning_transforms, joint_indices, joint_weights)
    return smpl_core.apply_global_transform(torch, v_posed, global_rotation, global_translation, rotation_type)
