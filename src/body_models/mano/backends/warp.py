"""Warp-accelerated MANO Torch backend."""

import torch
from jaxtyping import Float
from torch import Tensor

from body_models import common
from body_models.mano.backends import core
from body_models.mano.backends import torch as torch_backend
from body_models.mano.io import ManoWeights
from body_models.rotations import RotationType
from body_models.smpl.backends import core as smpl_core
from body_models.smpl.backends import warp as smpl_warp

forward_skeleton = torch_backend.forward_skeleton
prepare_identity = torch_backend.prepare_identity

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity"]


def forward_vertices(
    weights: ManoWeights,
    hand_pose: Float[Tensor, "B 15 N"] | Float[Tensor, "B 15 3 3"],
    wrist_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[Tensor, "*batch J 3"],
    local_joint_offsets: Float[Tensor, "*batch J 3"],
    rest_vertices: Float[Tensor, "*batch V 3"],
):
    if hand_pose.device.type != "cuda":
        return torch_backend.forward_vertices(
            weights=weights,
            hand_pose=hand_pose,
            wrist_rotation=wrist_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=rotation_type,
            rest_joints=rest_joints,
            local_joint_offsets=local_joint_offsets,
            rest_vertices=rest_vertices,
        )

    posedirs = weights.posedirs
    joint_indices = weights.lbs_joint_indices
    joint_weights = weights.lbs_joint_weights
    if vertex_indices is not None:
        selected_vertices = torch.as_tensor(vertex_indices, device=hand_pose.device)
        rest_vertices = rest_vertices[..., selected_vertices, :]
        posedirs = posedirs.reshape(posedirs.shape[0], -1, 3)[:, selected_vertices].reshape(posedirs.shape[0], -1)
        joint_indices = joint_indices[selected_vertices]
        joint_weights = joint_weights[selected_vertices]

    pose_matrices, T_world = core._forward_core(
        xp=torch,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        hand_mean=weights.hand_mean,
        local_joint_offsets=local_joint_offsets,
        hand_pose=hand_pose,
        wrist_rotation=wrist_rotation,
        rotation_type=rotation_type,
    )

    batch_shape = pose_matrices.shape[:-3]
    eye3 = common.eye_as(pose_matrices, batch_dims=(*batch_shape, 1), xp=torch)
    pose_delta = (pose_matrices[..., 1:, :, :] - eye3).reshape(*batch_shape, -1)
    v_shaped = rest_vertices + (pose_delta @ posedirs).reshape(*batch_shape, -1, 3)
    v_posed = smpl_warp.warp_linear_blend_skinning(v_shaped, rest_joints, T_world, joint_indices, joint_weights)
    return smpl_core.apply_global_transform(torch, v_posed, global_rotation, global_translation, rotation_type)
