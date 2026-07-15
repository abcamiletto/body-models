"""Warp-accelerated GarmentMeasurements Torch backend."""

import torch
from jaxtyping import Float
from nanomanifold import SE3, SO3
from torch import Tensor

from body_models.bodies.smpl.backends import warp as smpl_warp
from body_models.rotations import RotationType
from . import core
from ..io import GarmentMeasurementsWeights

__all__ = ["forward_vertices", "prepare_pose"]


def prepare_pose(
    weights: GarmentMeasurementsWeights,
    pose: Float[Tensor, "*batch J N"] | Float[Tensor, "*batch J 3 3"],
    rotation_type: RotationType = "axis_angle",
    *,
    bind_skeleton: Float[Tensor, "*batch J 7"],
    local_bind_translations: Float[Tensor, "*batch J 3"],
    skip_vertices: bool = False,
) -> core.GarmentMeasurementsPreparedPose:
    """Prepare pose transforms with fused Warp forward kinematics."""
    if any(value.dtype != torch.float32 for value in (pose, bind_skeleton, local_bind_translations)):
        raise TypeError("kernel='warp' supports only float32 tensors.")

    num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
    batch_shape = pose.shape[: -(num_rot_dims + 1)]
    num_joints = weights.bind_quats.shape[0]
    bind_skeleton = torch.broadcast_to(bind_skeleton, (*batch_shape, num_joints, 7))
    local_bind_translations = torch.broadcast_to(
        local_bind_translations,
        (*batch_shape, num_joints, 3),
    )

    pose_quats = SO3.convert(pose, src=rotation_type, dst="quat", xp=torch)
    local_quats = SO3.multiply(weights.bind_quats, pose_quats, xp=torch)
    local_rotations = SO3.convert(local_quats, src="quat", dst="rotmat", xp=torch)

    skeleton = smpl_warp.warp_forward_kinematics(local_rotations, local_bind_translations, weights.parents)

    prepared_pose: core.GarmentMeasurementsPreparedPose = {"skeleton_transforms": skeleton}
    if not skip_vertices:
        inverse_bind = SE3.to_matrix(SE3.inverse(bind_skeleton, xp=torch), xp=torch)
        prepared_pose["skinning_transforms"] = skeleton @ inverse_bind
    return prepared_pose


def forward_vertices(
    weights: GarmentMeasurementsWeights,
    rest_vertices: Float[Tensor, "*batch V 3"],
    skinning_transforms: Float[Tensor, "*batch J 4 4"],
    global_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
    global_translation: Float[Tensor, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    skin_weights = weights.skin_weights
    joint_indices = weights.skin_joint_indices
    joint_weights = weights.skin_joint_weights
    if vertex_indices is not None:
        rest_vertices = rest_vertices[..., vertex_indices, :]
        skin_weights = skin_weights[vertex_indices]
        joint_indices = joint_indices[vertex_indices]
        joint_weights = joint_weights[vertex_indices]

    vertices = smpl_warp.warp_affine_blend_skinning(
        rest_vertices,
        skinning_transforms,
        skin_weights,
        joint_indices,
        joint_weights,
    )
    return core.apply_global_transform(
        values=vertices,
        global_rotation=global_rotation,
        global_translation=global_translation,
        rotation_type=rotation_type,
        xp=torch,
    )
