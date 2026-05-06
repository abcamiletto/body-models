"""PyTorch SMPL backend."""

import torch
from jaxtyping import Float
from torch import Tensor

from body_models.rotations import RotationType
from body_models.smpl.backends.core import forward_skeleton as _forward_skeleton
from body_models.smpl.backends.core import forward_vertices as _forward_vertices
from body_models.smpl.io import SmplWeights

__all__ = ["forward_vertices", "forward_skeleton"]


def forward_vertices(
    weights: SmplWeights,
    shape: Float[Tensor, "B 10"],
    body_pose: Float[Tensor, "B 23 N"] | Float[Tensor, "B 23 3 3"],
    pelvis_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return _forward_vertices(
        v_template=weights.v_template,
        shapedirs=weights.shapedirs,
        posedirs=weights.posedirs,
        lbs_weights=weights.lbs_weights,
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        shape=shape,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        xp=torch,
    )


def forward_skeleton(
    weights: SmplWeights,
    shape: Float[Tensor, "B 10"],
    body_pose: Float[Tensor, "B 23 N"] | Float[Tensor, "B 23 3 3"],
    pelvis_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return _forward_skeleton(
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        shape=shape,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        xp=torch,
    )
