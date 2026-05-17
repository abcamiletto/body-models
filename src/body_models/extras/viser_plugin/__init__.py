"""Optional ``viser`` helpers for body model visualization."""

from importlib.util import find_spec

if find_spec("viser") is None:
    raise ImportError(
        "body_models.extras.viser_plugin requires the optional 'viser' dependency. "
        "Install it with `pip install 'body-models[viser]'` or `uv add 'body-models[viser]'`."
    )

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
