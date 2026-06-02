"""PyTorch SMPL-X backend."""

import torch
from torch import Tensor
from jaxtyping import Float

from body_models.rotations import RotationType
from body_models.smplx.backends.core import SmplxIdentity, SmplxPreparedPose
from body_models.smplx.backends.core import forward_skeleton as _forward_skeleton
from body_models.smplx.backends.core import forward_vertices as _forward_vertices
from body_models.smplx.backends.core import prepare_identity as _prepare_identity
from body_models.smplx.backends.core import prepare_pose as _prepare_pose
from body_models.smplx.io import SmplxWeights

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def prepare_identity(
    weights: SmplxWeights,
    shape: Float[Tensor, "*batch 10"],
    expression: Float[Tensor, "*batch 10"],
    skip_vertices: bool = False,
) -> SmplxIdentity:
    return _prepare_identity(
        xp=torch,
        v_template=weights.v_template,
        shapedirs=weights.shapedirs,
        exprdirs=weights.exprdirs,
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        j_exprdirs=weights.j_exprdirs,
        parents=weights.parents,
        shape=shape,
        expression=expression,
        skip_vertices=skip_vertices,
    )


def prepare_pose(
    weights: SmplxWeights,
    body_pose: Float[Tensor, "*batch 21 N"] | Float[Tensor, "*batch 21 3 3"],
    hand_pose: Float[Tensor, "*batch 30 N"] | Float[Tensor, "*batch 30 3 3"],
    head_pose: Float[Tensor, "*batch 3 N"] | Float[Tensor, "*batch 3 3 3"],
    pelvis_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    local_joint_offsets: Float[Tensor, "*batch J 3"],
    rest_joints: Float[Tensor, "*batch J 3"],
    skip_vertices: bool = False,
) -> SmplxPreparedPose:
    """Precompute pose-dependent state for repeated forward passes."""
    return _prepare_pose(
        xp=torch,
        posedirs=weights.posedirs,
        kinematic_fronts=weights.kinematic_fronts,
        hand_mean=weights.hand_mean,
        body_pose=body_pose,
        hand_pose=hand_pose,
        head_pose=head_pose,
        pelvis_rotation=pelvis_rotation,
        rotation_type=rotation_type,
        local_joint_offsets=local_joint_offsets,
        rest_joints=rest_joints,
        skip_vertices=skip_vertices,
    )


def forward_vertices(
    weights: SmplxWeights,
    rest_vertices: Float[Tensor, "*batch V 3"],
    skinning_transforms: Float[Tensor, "*batch J 4 4"],
    pose_offsets: Float[Tensor, "*batch V 3"],
    global_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
    global_translation: Float[Tensor, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return _forward_vertices(
        lbs_weights=weights.lbs_weights,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        rest_vertices=rest_vertices,
        skinning_transforms=skinning_transforms,
        pose_offsets=pose_offsets,
        xp=torch,
    )


def forward_skeleton(
    weights: SmplxWeights,
    skeleton_transforms: Float[Tensor, "*batch J 4 4"],
    global_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
    global_translation: Float[Tensor, "*batch 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return _forward_skeleton(
        parents=weights.parents,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        skeleton_transforms=skeleton_transforms,
        xp=torch,
    )
