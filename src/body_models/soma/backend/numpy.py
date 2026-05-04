"""NumPy SOMA backend."""

from . import core

__all__ = [
    "apply_pose_correctives",
    "fit_rigid_transform",
    "forward_skeleton",
    "forward_vertices",
    "linear_blend_skinning",
    "prepare_data",
    "prepare_identity",
    "prepare_identity_model",
    "prepare_identity_transfer",
]

apply_pose_correctives = core.apply_pose_correctives
fit_rigid_transform = core.fit_rigid_transform
forward_skeleton = core.forward_skeleton
forward_vertices = core.forward_vertices
linear_blend_skinning = core.linear_blend_skinning
prepare_data = core.prepare_data
prepare_identity = core.prepare_identity
prepare_identity_transfer = core.prepare_identity_transfer


def prepare_identity_model(model_type: str, identity_model):
    if model_type == "mhr":
        from ...mhr.numpy import MHR

        return MHR(model_path=identity_model.model_path, simplify=1.0)
    return identity_model
