"""NumPy SOMA backend."""

from .. import identities
from . import core

__all__ = [
    "apply_pose_correctives",
    "fit_rigid_transform",
    "forward_skeleton",
    "forward_vertices",
    "linear_blend_skinning",
    "PreparedSomaIdentity",
    "prepare_data",
    "prepare_identity_backend",
]

apply_pose_correctives = core.apply_pose_correctives
fit_rigid_transform = core.fit_rigid_transform
forward_skeleton = core.forward_skeleton
forward_vertices = core.forward_vertices
linear_blend_skinning = core.linear_blend_skinning
PreparedSomaIdentity = core.PreparedSomaIdentity
prepare_data = core.prepare_data


def prepare_identity_backend(identity_backend: identities.IdentityBackend) -> identities.IdentityBackend:
    return identities.prepare_backend(identity_backend, "numpy")
