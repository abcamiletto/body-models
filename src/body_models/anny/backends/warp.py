"""PyTorch ANNY backend."""

import torch
from torch import Tensor
from jaxtyping import Float

from body_models.anny.backends.core import AnnyIdentity, AnnyPreparedPose
from body_models.anny.backends.core import forward_skeleton as _forward_skeleton
from body_models.anny.backends.core import forward_vertices as _forward_vertices
from body_models.anny.backends.core import prepare_identity as _prepare_identity
from body_models.anny.backends.core import prepare_pose as _prepare_pose
from body_models.anny.io import AnnyWeights
from body_models.rotations import RotationType

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def prepare_identity(
    weights: AnnyWeights,
    shape: Float[Tensor, "*batch 6"],
    extrapolate_phenotypes: bool = False,
    skip_vertices: bool = False,
) -> AnnyIdentity:
    return _prepare_identity(
        xp=torch,
        template_vertices=weights.template_vertices,
        blendshapes=weights.blendshapes,
        template_bone_heads=weights.template_bone_heads,
        template_bone_tails=weights.template_bone_tails,
        bone_heads_blendshapes=weights.bone_heads_blendshapes,
        bone_tails_blendshapes=weights.bone_tails_blendshapes,
        bone_rolls_rotmat=weights.bone_rolls_rotmat,
        phenotype_mask=weights.phenotype_mask,
        anchors=weights.anchors,
        y_axis=weights.y_axis,
        degenerate_rotation=weights.degenerate_rotation,
        extrapolate_phenotypes=extrapolate_phenotypes,
        shape=shape,
        skip_vertices=skip_vertices,
    )


def prepare_pose(
    weights: AnnyWeights,
    pose: Float[Tensor, "*batch J N"] | Float[Tensor, "*batch J 3 3"],
    rotation_type: RotationType = "axis_angle",
    *,
    rest_skeleton_transforms: Float[Tensor, "*batch J 4 4"],
    skip_vertices: bool = False,
) -> AnnyPreparedPose:
    """Precompute pose-dependent state for repeated forward passes."""
    return _prepare_pose(
        xp=torch,
        kinematic_fronts=weights.kinematic_fronts,
        pose=pose,
        rotation_type=rotation_type,
        rest_skeleton_transforms=rest_skeleton_transforms,
        skip_vertices=skip_vertices,
    )


def forward_vertices(
    weights: AnnyWeights,
    rest_vertices: Float[Tensor, "*batch V 3"],
    skinning_transforms: Float[Tensor, "*batch J 4 4"],
    global_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
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
        xp=torch,
    )


def forward_skeleton(
    weights: AnnyWeights,
    skeleton_transforms: Float[Tensor, "*batch J 4 4"],
    global_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return _forward_skeleton(
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        skeleton_transforms=skeleton_transforms,
        xp=torch,
    )
