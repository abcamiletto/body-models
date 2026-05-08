"""PyTorch ANNY backend."""

import torch
from jaxtyping import Float
from torch import Tensor

from body_models.anny.backends.core import forward_skeleton as _forward_skeleton
from body_models.anny.backends.core import forward_vertices as _forward_vertices
from body_models.anny.io import AnnyWeights
from body_models.rotations import RotationType

__all__ = ["forward_vertices", "forward_skeleton"]


def forward_vertices(
    weights: AnnyWeights,
    gender: Float[Tensor, "B"],
    age: Float[Tensor, "B"],
    muscle: Float[Tensor, "B"],
    weight: Float[Tensor, "B"],
    height: Float[Tensor, "B"],
    proportions: Float[Tensor, "B"],
    pose: Float[Tensor, "B J N"] | Float[Tensor, "B J 3 3"],
    global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    extrapolate_phenotypes: bool = False,
):
    return _forward_vertices(
        template_vertices=weights.template_vertices,
        blendshapes=weights.blendshapes,
        template_bone_heads=weights.template_bone_heads,
        template_bone_tails=weights.template_bone_tails,
        bone_heads_blendshapes=weights.bone_heads_blendshapes,
        bone_tails_blendshapes=weights.bone_tails_blendshapes,
        bone_rolls_rotmat=weights.bone_rolls_rotmat,
        lbs_weights=weights.lbs_weights,
        phenotype_mask=weights.phenotype_mask,
        anchors=weights.anchors,
        kinematic_fronts=weights.kinematic_fronts,
        y_axis=weights.y_axis,
        degenerate_rotation=weights.degenerate_rotation,
        extrapolate_phenotypes=extrapolate_phenotypes,
        gender=gender,
        age=age,
        muscle=muscle,
        weight=weight,
        height=height,
        proportions=proportions,
        pose=pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        xp=torch,
    )


def forward_skeleton(
    weights: AnnyWeights,
    gender: Float[Tensor, "B"],
    age: Float[Tensor, "B"],
    muscle: Float[Tensor, "B"],
    weight: Float[Tensor, "B"],
    height: Float[Tensor, "B"],
    proportions: Float[Tensor, "B"],
    pose: Float[Tensor, "B J N"] | Float[Tensor, "B J 3 3"],
    global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    extrapolate_phenotypes: bool = False,
):
    return _forward_skeleton(
        template_bone_heads=weights.template_bone_heads,
        template_bone_tails=weights.template_bone_tails,
        bone_heads_blendshapes=weights.bone_heads_blendshapes,
        bone_tails_blendshapes=weights.bone_tails_blendshapes,
        bone_rolls_rotmat=weights.bone_rolls_rotmat,
        phenotype_mask=weights.phenotype_mask,
        anchors=weights.anchors,
        kinematic_fronts=weights.kinematic_fronts,
        y_axis=weights.y_axis,
        degenerate_rotation=weights.degenerate_rotation,
        extrapolate_phenotypes=extrapolate_phenotypes,
        gender=gender,
        age=age,
        muscle=muscle,
        weight=weight,
        height=height,
        proportions=proportions,
        pose=pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        xp=torch,
    )
