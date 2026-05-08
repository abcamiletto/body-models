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
    "PreparedSomaIdentity",
]

fit_rigid_transform = core.fit_rigid_transform
forward_skeleton = torch_backend.forward_skeleton
apply_pose_correctives = torch_backend.apply_pose_correctives
PreparedSomaIdentity = core.PreparedSomaIdentity


def forward_vertices(*args, **kwargs):
    data = kwargs["data"]
    return core._forward_vertices_with(
        *args,
        **kwargs,
        apply_pose_correctives_fn=apply_pose_correctives,
        linear_blend_skinning_fn=_linear_blend_skinning(data),
    )


def _linear_blend_skinning(data):
    def skin(xp, bind_shape, skin_weights, bone_transforms):
        return linear_blend_skinning(
            xp,
            bind_shape,
            skin_weights,
            bone_transforms,
            joint_indices=data.skin_joint_indices_active,
            joint_weights=data.skin_joint_weights_active,
        )

    return skin


def linear_blend_skinning(xp, bind_shape, skin_weights, bone_transforms, *, joint_indices, joint_weights):
    if bind_shape.device.type != "cuda":
        return torch_backend.linear_blend_skinning(xp, bind_shape, skin_weights, bone_transforms)
    return smpl_warp.warp_affine_blend_skinning(bind_shape, bone_transforms, joint_indices, joint_weights)
