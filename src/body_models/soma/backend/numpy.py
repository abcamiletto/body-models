"""NumPy SOMA backend."""

from .. import identities
from . import core

__all__ = [
    "apply_pose_correctives",
    "fit_rigid_transform",
    "forward_skeleton",
    "forward_vertices",
    "linear_blend_skinning",
    "prepare_data",
    "prepare_identity",
    "prepare_identity_backend",
]

apply_pose_correctives = core.apply_pose_correctives
fit_rigid_transform = core.fit_rigid_transform
forward_skeleton = core.forward_skeleton
forward_vertices = core.forward_vertices
linear_blend_skinning = core.linear_blend_skinning
prepare_data = core.prepare_data
prepare_identity = core.prepare_identity


def prepare_identity_backend(identity_backend: identities.IdentityBackend) -> identities.IdentityBackend:
    if not isinstance(identity_backend, identities.TransferredIdentityBackend):
        return identity_backend
    if identity_backend.model_type != "mhr":
        return identity_backend

    return identities.TransferredIdentityBackend(
        model_type=identity_backend.model_type,
        identity_dim=identity_backend.identity_dim,
        num_scale_params=identity_backend.num_scale_params,
        default_identity_value=identity_backend.default_identity_value,
        model=_load_mhr_model(identity_backend.model),
        transfer=identity_backend.transfer,
    )


def _load_mhr_model(identity_model):
    from ...mhr.numpy import MHR

    return MHR(model_path=identity_model.model_path, simplify=1.0)
