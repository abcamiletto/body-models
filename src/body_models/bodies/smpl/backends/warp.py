"""Warp-accelerated SMPL Torch backend."""

import torch
from jaxtyping import Float
from torch import Tensor

from body_models.common import skinning
from body_models.common import warp as warp_backend
from body_models.rotations import RotationType
from body_models.bodies.smpl.backends import core
from body_models.bodies.smpl.backends import torch as torch_backend
from body_models.bodies.smpl.io import SmplWeights

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def prepare_identity(
    weights: SmplWeights,
    shape: Float[Tensor, "*batch 10"],
    skip_vertices: bool = False,
) -> core.SmplIdentity:
    return torch_backend.prepare_identity(weights, shape, skip_vertices=skip_vertices)


def prepare_pose(
    weights: SmplWeights,
    body_pose: Float[Tensor, "*batch 23 N"] | Float[Tensor, "*batch 23 3 3"],
    pelvis_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    local_joint_offsets: Float[Tensor, "*batch J 3"],
    rest_joints: Float[Tensor, "*batch J 3"],
    skip_vertices: bool = False,
) -> core.SmplPreparedPose:
    """Precompute pose-dependent state for repeated forward passes."""
    return torch_backend.prepare_pose(
        weights,
        body_pose,
        pelvis_rotation,
        rotation_type=rotation_type,
        local_joint_offsets=local_joint_offsets,
        rest_joints=rest_joints,
        skip_vertices=skip_vertices,
    )


def forward_vertices(
    weights: SmplWeights,
    rest_vertices: Float[Tensor, "*batch V 3"],
    skinning_transforms: Float[Tensor, "*batch J 4 4"],
    pose_offsets: Float[Tensor, "*batch V 3"],
    global_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
    global_translation: Float[Tensor, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
) -> Float[Tensor, "*batch V 3"]:
    joint_indices = weights.lbs_joint_indices
    joint_weights = weights.lbs_joint_weights
    if vertex_indices is not None:
        rest_vertices = rest_vertices[..., vertex_indices, :]
        pose_offsets = pose_offsets[..., vertex_indices, :]
        joint_indices = joint_indices[vertex_indices]
        joint_weights = joint_weights[vertex_indices]

    vertices = rest_vertices + pose_offsets
    vertices = warp_backend.compact_linear_blend_skinning(
        vertices,
        skinning_transforms,
        joint_indices=joint_indices,
        joint_weights=joint_weights,
    )
    return skinning.apply_global_transform(
        vertices,
        global_rotation,
        global_translation,
        rotation_type,
        xp=torch,
    )


def forward_skeleton(
    weights: SmplWeights,
    skeleton_transforms: Float[Tensor, "*batch J 4 4"],
    global_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
    global_translation: Float[Tensor, "*batch 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
) -> Float[Tensor, "*batch J 4 4"]:
    return core.forward_skeleton(
        parents=weights.parents,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        skeleton_transforms=skeleton_transforms,
        xp=torch,
    )
