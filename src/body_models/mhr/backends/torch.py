"""PyTorch MHR backend."""

import torch
from torch import Tensor
from jaxtyping import Float

from body_models.mhr.backends.core import MhrIdentity, MhrPreparedPose
from body_models.mhr.backends.core import forward_skeleton as _forward_skeleton
from body_models.mhr.backends.core import forward_vertices as _forward_vertices
from body_models.mhr.backends.core import prepare_identity as _prepare_identity
from body_models.mhr.backends.core import prepare_pose as _prepare_pose
from body_models.mhr.io import MhrWeights

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def prepare_identity(
    weights: MhrWeights,
    shape: Float[Tensor, "*batch 45"],
    expression: Float[Tensor, "*batch 72"],
    skip_vertices: bool = False,
) -> MhrIdentity:
    return _prepare_identity(
        xp=torch,
        base_vertices=weights.base_vertices,
        blendshape_dirs=weights.blendshape_dirs,
        shape=shape,
        expression=expression,
        skip_vertices=skip_vertices,
    )


def prepare_pose(
    weights: MhrWeights, pose: Float[Tensor, "*batch 204"], skip_vertices: bool = False
) -> MhrPreparedPose:
    """Precompute pose-dependent state for repeated forward passes."""
    return _prepare_pose(
        xp=torch,
        joint_offsets=weights.joint_offsets,
        joint_pre_rotations=weights.joint_pre_rotations,
        parameter_transform=weights.parameter_transform,
        kinematic_fronts=weights.kinematic_fronts,
        num_joints=len(weights.parents),
        shape_dim=45,
        pose=pose,
        skip_vertices=skip_vertices,
    )


def forward_vertices(
    weights: MhrWeights,
    rest_vertices: Float[Tensor, "*batch V 3"],
    joint_translations: Float[Tensor, "*batch J 3"],
    joint_rotations: Float[Tensor, "*batch J 3 3"],
    joint_scales: Float[Tensor, "*batch J 1"],
    joint_params: Float[Tensor, "*batch J 7"],
    global_rotation: Float[Tensor, "B 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
):
    return _forward_vertices(
        base_vertices=weights.base_vertices,
        blendshape_dirs=weights.blendshape_dirs,
        skin_weights=weights.skin_weights,
        skin_indices=weights.skin_indices,
        joint_offsets=weights.joint_offsets,
        joint_pre_rotations=weights.joint_pre_rotations,
        parameter_transform=weights.parameter_transform,
        bind_inv_linear=weights.bind_inv_linear,
        bind_inv_translation=weights.bind_inv_translation,
        expr_dim=72,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        corrective_W1=weights.corrective_W1,
        corrective_W2=weights.corrective_W2,
        rest_vertices=rest_vertices,
        joint_translations=joint_translations,
        joint_rotations=joint_rotations,
        joint_scales=joint_scales,
        joint_params=joint_params,
        xp=torch,
    )


def forward_skeleton(
    weights: MhrWeights,
    *,
    joint_translations: Float[Tensor, "*batch J 3"],
    joint_rotations: Float[Tensor, "*batch J 3 3"],
    joint_scales: Float[Tensor, "*batch J 1"],
    global_rotation: Float[Tensor, "B 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
):
    return _forward_skeleton(
        num_joints=len(weights.parents),
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        joint_translations=joint_translations,
        joint_rotations=joint_rotations,
        joint_scales=joint_scales,
        xp=torch,
    )
