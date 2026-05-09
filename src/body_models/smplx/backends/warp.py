"""Warp-accelerated SMPL-X Torch backend."""

import torch
from jaxtyping import Float
from torch import Tensor

from body_models import common
from body_models.rotations import RotationType, is_rotmat_type
from body_models.smpl.backends import core as smpl_core
from body_models.smpl.backends import warp as smpl_warp
from body_models.smplx.backends import core
from body_models.smplx.backends import torch as torch_backend
from body_models.smplx.io import SmplxWeights

forward_skeleton = torch_backend.forward_skeleton

__all__ = ["forward_vertices", "forward_skeleton"]


def forward_vertices(
    weights: SmplxWeights,
    shape: Float[Tensor, "B 10"],
    body_pose: Float[Tensor, "B 21 N"] | Float[Tensor, "B 21 3 3"],
    hand_pose: Float[Tensor, "B 30 N"] | Float[Tensor, "B 30 3 3"],
    head_pose: Float[Tensor, "B 3 N"] | Float[Tensor, "B 3 3 3"],
    expression: Float[Tensor, "B 10"] | None = None,
    pelvis_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    if shape.device.type != "cuda":
        return torch_backend.forward_vertices(
            weights=weights,
            shape=shape,
            body_pose=body_pose,
            hand_pose=hand_pose,
            head_pose=head_pose,
            expression=expression,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=rotation_type,
        )

    v_template = weights.v_template
    shapedirs = weights.shapedirs
    exprdirs = weights.exprdirs
    posedirs = weights.posedirs
    joint_indices = weights.lbs_joint_indices
    joint_weights = weights.lbs_joint_weights
    if vertex_indices is not None:
        selected_vertices = torch.as_tensor(vertex_indices, device=shape.device)
        v_template = v_template[selected_vertices]
        shapedirs = shapedirs[selected_vertices]
        exprdirs = exprdirs[selected_vertices]
        posedirs = posedirs.reshape(posedirs.shape[0], -1, 3)[:, selected_vertices].reshape(posedirs.shape[0], -1)
        joint_indices = joint_indices[selected_vertices]
        joint_weights = joint_weights[selected_vertices]
    if expression is None:
        pose_ndim = 3 if is_rotmat_type(rotation_type) else 2
        batch_shape = body_pose.shape[:-pose_ndim]
        expression = torch.zeros((*batch_shape, 10), dtype=shape.dtype, device=shape.device)

    v_t, j_t, pose_matrices, T_world = core._forward_core(
        xp=torch,
        v_template=v_template,
        shapedirs=shapedirs,
        exprdirs=exprdirs,
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        j_exprdirs=weights.j_exprdirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        hand_mean=weights.hand_mean,
        shape=shape,
        expression=expression,
        body_pose=body_pose,
        hand_pose=hand_pose,
        head_pose=head_pose,
        pelvis_rotation=pelvis_rotation,
        skeleton_only=False,
        rotation_type=rotation_type,
    )

    batch_shape = pose_matrices.shape[:-3]
    eye3 = common.eye_as(pose_matrices, batch_dims=(*batch_shape, 1), xp=torch)
    pose_delta = (pose_matrices[..., 1:, :, :] - eye3).reshape(*batch_shape, -1)
    v_shaped = v_t + (pose_delta @ posedirs).reshape(*batch_shape, -1, 3)
    v_posed = smpl_warp.warp_linear_blend_skinning(v_shaped, j_t, T_world, joint_indices, joint_weights)
    return smpl_core.apply_global_transform(torch, v_posed, global_rotation, global_translation, rotation_type)
