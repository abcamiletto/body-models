"""PyTorch FLAME backend."""

import torch
from torch import Tensor
from jaxtyping import Float

from body_models.flame.backends.core import FlameIdentity, FlamePreparedPose
from body_models.flame.backends.core import forward_skeleton as _forward_skeleton
from body_models.flame.backends.core import forward_vertices as _forward_vertices
from body_models.flame.backends.core import prepare_identity as _prepare_identity
from body_models.flame.backends.core import prepare_pose as _prepare_pose
from body_models.flame.io import FlameWeights
from body_models.rotations import RotationType

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def prepare_identity(
    weights: FlameWeights,
    shape: Float[Tensor, "*batch S"],
    expression: Float[Tensor, "*batch E"],
    skip_vertices: bool = False,
) -> FlameIdentity:
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
    weights: FlameWeights,
    pose: Float[Tensor, "B 4 N"] | Float[Tensor, "B 4 3 3"],
    head_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    local_joint_offsets: Float[Tensor, "*batch J 3"],
    skip_vertices: bool = False,
) -> FlamePreparedPose:
    """Precompute pose-dependent state for repeated forward passes."""
    return _prepare_pose(
        xp=torch,
        posedirs=weights.posedirs,
        kinematic_fronts=weights.kinematic_fronts,
        pose=pose,
        head_rotation=head_rotation,
        rotation_type=rotation_type,
        local_joint_offsets=local_joint_offsets,
        skip_vertices=skip_vertices,
    )


def forward_vertices(
    weights: FlameWeights,
    global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[Tensor, "*batch J 3"],
    rest_vertices: Float[Tensor, "*batch V 3"],
    joint_transforms: Float[Tensor, "*batch J 4 4"],
    pose_offsets: Float[Tensor, "*batch V 3"],
):
    return _forward_vertices(
        lbs_weights=weights.lbs_weights,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        rest_joints=rest_joints,
        rest_vertices=rest_vertices,
        joint_transforms=joint_transforms,
        pose_offsets=pose_offsets,
        xp=torch,
    )


def forward_skeleton(
    weights: FlameWeights,
    global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    joint_transforms: Float[Tensor, "*batch J 4 4"],
):
    return _forward_skeleton(
        parents=weights.parents,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        joint_transforms=joint_transforms,
        xp=torch,
    )
