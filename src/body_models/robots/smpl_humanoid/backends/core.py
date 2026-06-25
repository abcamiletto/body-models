"""Backend-agnostic SMPL humanoid rigid articulated model computation."""

from body_models.robots.g1.backends.core import (  # noqa: F401
    GLOBAL_ROTATION_TYPES,
    VALID_ROTATION_TYPES,
    Convention,
    RotationType,
    forward_links,
    forward_skeleton,
    forward_vertices,
    joint_meshes,
    link_mesh,
)

SKIN_WEIGHTS_ERROR = "SmplHumanoid is a rigid articulated model and does not define skin_weights."

__all__ = [
    "GLOBAL_ROTATION_TYPES",
    "SKIN_WEIGHTS_ERROR",
    "VALID_ROTATION_TYPES",
    "Convention",
    "RotationType",
    "forward_links",
    "forward_skeleton",
    "forward_vertices",
    "joint_meshes",
    "link_mesh",
]
