"""Warp-accelerated ANNY Torch backend."""

import torch
from jaxtyping import Float
from torch import Tensor

from body_models.anny.backends import core
from body_models.anny.backends import torch as torch_backend
from body_models.anny.io import AnnyWeights
from body_models.rotations import RotationType
from body_models.smpl.backends import warp as smpl_warp

prepare_identity = torch_backend.prepare_identity

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity"]


def forward_vertices(
    weights: AnnyWeights,
    pose: Float[Tensor, "*batch J N"] | Float[Tensor, "*batch J 3 3"],
    global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    extrapolate_phenotypes: bool = False,
    *,
    rest_bone_poses: Float[Tensor, "*batch J 4 4"],
    rest_vertices: Float[Tensor, "*batch V 3"],
):
    if pose.device.type != "cuda":
        return torch_backend.forward_vertices(
            weights=weights,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=rotation_type,
            extrapolate_phenotypes=extrapolate_phenotypes,
            rest_bone_poses=rest_bone_poses,
            rest_vertices=rest_vertices,
        )

    rest_vertices, bone_transforms = core.forward_unskinned_vertices(
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
        pose=pose,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        rest_bone_poses=rest_bone_poses,
        rest_vertices=rest_vertices,
        xp=torch,
    )
    joint_indices = weights.lbs_joint_indices
    joint_weights = weights.lbs_joint_weights
    if vertex_indices is not None:
        joint_indices = joint_indices[vertex_indices]
        joint_weights = joint_weights[vertex_indices]

    vertices = smpl_warp.warp_affine_blend_skinning(rest_vertices, bone_transforms, joint_indices, joint_weights)
    return core.apply_global_transform(torch, vertices, global_rotation, global_translation, rotation_type)


def forward_skeleton(
    weights: AnnyWeights,
    pose: Float[Tensor, "*batch J N"] | Float[Tensor, "*batch J 3 3"],
    global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    extrapolate_phenotypes: bool = False,
    *,
    rest_bone_poses: Float[Tensor, "*batch J 4 4"],
    rest_vertices: Float[Tensor, "*batch V 3"] | None = None,
):
    return torch_backend.forward_skeleton(
        weights=weights,
        pose=pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        extrapolate_phenotypes=extrapolate_phenotypes,
        rest_bone_poses=rest_bone_poses,
        rest_vertices=rest_vertices,
    )
