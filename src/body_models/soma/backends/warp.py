"""Warp-accelerated SOMA Torch backend."""

from body_models.smpl.backends import warp as smpl_warp
from body_models.soma.backends import core
from body_models.soma.backends import torch as torch_backend

__all__ = [
    "apply_pose_correctives",
    "fit_rigid_transform",
    "forward_skeleton",
    "forward_vertices",
    "linear_blend_skinning",
    "SomaIdentity",
    "SomaPreparedPose",
    "prepare_pose",
    "prepare_identity_from_rest_shape",
]

fit_rigid_transform = core.fit_rigid_transform
forward_skeleton = torch_backend.forward_skeleton
apply_pose_correctives = torch_backend.apply_pose_correctives
SomaIdentity = core.SomaIdentity
SomaPreparedPose = core.SomaPreparedPose
prepare_pose = core.prepare_pose


def prepare_identity_from_rest_shape(*args, **kwargs):
    return core.prepare_identity_from_rest_shape(
        *args,
        **kwargs,
        linear_blend_skinning_fn=torch_backend.linear_blend_skinning,
    )


def forward_vertices(*args, **kwargs):
    data = kwargs["data"]
    return core._forward_vertices_with(
        *args,
        **kwargs,
        apply_pose_correctives_fn=apply_pose_correctives,
        linear_blend_skinning_fn=_linear_blend_skinning(data),
    )


def _linear_blend_skinning(data):
    def skin(xp, bind_shape, skin_weights, skinning_transforms):
        return linear_blend_skinning(
            xp,
            bind_shape,
            skin_weights,
            skinning_transforms,
            joint_indices=data.skin_joint_indices_active,
            joint_weights=data.skin_joint_weights_active,
        )

    return skin


def linear_blend_skinning(xp, bind_shape, skin_weights, skinning_transforms, *, joint_indices, joint_weights):
    if bind_shape.device.type != "cuda":
        return torch_backend.linear_blend_skinning(xp, bind_shape, skin_weights, skinning_transforms)
    return smpl_warp.warp_affine_blend_skinning(bind_shape, skinning_transforms, joint_indices, joint_weights)
