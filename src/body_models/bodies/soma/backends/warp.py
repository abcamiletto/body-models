"""Warp-accelerated SOMA Torch backend."""

from body_models.common import warp as warp_backend
from body_models.bodies.soma.backends import core
from body_models.bodies.soma.backends import torch as torch_backend

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
prepare_pose = torch_backend.prepare_pose


def prepare_identity_from_rest_shape(*args, **kwargs):
    return core.prepare_identity_from_rest_shape(
        *args,
        **kwargs,
        linear_blend_skinning_fn=torch_backend.linear_blend_skinning,
    )


def forward_vertices(*args, **kwargs):
    data = kwargs["data"]
    vertex_indices = kwargs.get("vertex_indices")
    return core._forward_vertices_with(
        *args,
        **kwargs,
        linear_blend_skinning_fn=_linear_blend_skinning(data, vertex_indices),
    )


def _linear_blend_skinning(data, vertex_indices):
    # SOMA's compact weights include the unskinned world root, while prepared
    # skinning transforms and the dense Torch weights omit it.
    joint_indices = data.skin_joint_indices_active - 1
    joint_weights = data.skin_joint_weights_active
    if vertex_indices is not None:
        joint_indices = joint_indices[vertex_indices]
        joint_weights = joint_weights[vertex_indices]

    def skin(xp, bind_shape, skin_weights, skinning_transforms):
        return linear_blend_skinning(
            xp,
            bind_shape,
            skin_weights,
            skinning_transforms,
            joint_indices=joint_indices,
            joint_weights=joint_weights,
        )

    return skin


def linear_blend_skinning(
    xp,
    bind_shape,
    skin_weights,
    skinning_transforms,
    *,
    joint_indices,
    joint_weights,
):
    return warp_backend.compact_linear_blend_skinning(
        bind_shape,
        skinning_transforms,
        joint_indices=joint_indices,
        joint_weights=joint_weights,
    )
