"""Optional ``viser`` helpers for body model visualization."""

from body_models.extras.viser_plugin.body_model import ViserBodyModelHandle, add_body_model
from body_models.extras.viser_plugin.rigid_body import ViserRigidBodyModelHandle, add_rigid_body_model
from body_models.extras.viser_plugin.skeleton import ViserSkeletonHandle, add_skeleton

__all__ = [
    "ViserBodyModelHandle",
    "ViserRigidBodyModelHandle",
    "ViserSkeletonHandle",
    "add_body_model",
    "add_rigid_body_model",
    "add_skeleton",
]
