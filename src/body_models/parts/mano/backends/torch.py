"""PyTorch MANO backend."""

import torch
from torch import Tensor
from jaxtyping import Float

from body_models.parts.mano.backends.core import ManoIdentity, ManoPreparedPose
from body_models.parts.mano.backends.core import forward_skeleton as _forward_skeleton
from body_models.parts.mano.backends.core import forward_vertices as _forward_vertices
from body_models.parts.mano.backends.core import prepare_identity as _prepare_identity
from body_models.parts.mano.backends.core import prepare_pose as _prepare_pose
from body_models.parts.mano.io import ManoWeights
from body_models.rotations import RotationType

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def prepare_identity(
    weights: ManoWeights,
    shape: Float[Tensor, "*batch 10"],
    skip_vertices: bool = False,
) -> ManoIdentity:
    return _prepare_identity(
        xp=torch,
        v_template=weights.v_template,
        shapedirs=weights.shapedirs,
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        shape=shape,
        skip_vertices=skip_vertices,
    )


def prepare_pose(
    weights: ManoWeights,
    hand_pose: Float[Tensor, "B 15 N"] | Float[Tensor, "B 15 3 3"],
    wrist_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    local_joint_offsets: Float[Tensor, "*batch J 3"],
    rest_joints: Float[Tensor, "*batch J 3"],
    skip_vertices: bool = False,
) -> ManoPreparedPose:
    """Precompute pose-dependent state for repeated forward passes."""
    return _prepare_pose(
        xp=torch,
        posedirs=weights.posedirs,
        kinematic_fronts=weights.kinematic_fronts,
        hand_mean=weights.hand_mean,
        hand_pose=hand_pose,
        wrist_rotation=wrist_rotation,
        rotation_type=rotation_type,
        local_joint_offsets=local_joint_offsets,
        rest_joints=rest_joints,
        skip_vertices=skip_vertices,
    )


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
    weights: ManoWeights,
    skeleton_transforms: Float[Tensor, "*batch J 4 4"],
    global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
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
