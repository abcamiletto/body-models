"""Warp-accelerated MHR Torch backend."""

import torch
from jaxtyping import Float
from torch import Tensor

from body_models.common import skinning
from body_models.common import warp as warp_backend
from body_models.bodies.mhr.backends import torch as torch_backend
from body_models.bodies.mhr.io import MhrWeights

forward_skeleton = torch_backend.forward_skeleton
prepare_identity = torch_backend.prepare_identity
prepare_pose = torch_backend.prepare_pose

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def forward_vertices(
    weights: MhrWeights,
    rest_vertices: Float[Tensor, "*batch V 3"],
    skinning_transforms: Float[Tensor, "*batch J 4 4"],
    pose_offsets: Float[Tensor, "*batch V 3"],
    global_rotation: Float[Tensor, "*batch 3"] | None = None,
    global_translation: Float[Tensor, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
) -> Float[Tensor, "*batch V 3"]:
    joint_indices = weights.skin_indices
    joint_weights = weights.skin_weights
    if vertex_indices is not None:
        rest_vertices = rest_vertices[..., vertex_indices, :]
        pose_offsets = pose_offsets[..., vertex_indices, :]
        joint_indices = joint_indices[vertex_indices]
        joint_weights = joint_weights[vertex_indices]

    vertices = warp_backend.compact_linear_blend_skinning(
        rest_vertices + pose_offsets,
        skinning_transforms,
        joint_indices=joint_indices,
        joint_weights=joint_weights,
    )
    return skinning.apply_global_transform(
        vertices,
        global_rotation,
        global_translation,
        xp=torch,
    )
