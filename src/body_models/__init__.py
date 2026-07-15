"""Multi-runtime parametric and articulated body models."""

from importlib import import_module
import sys

from body_models.base import RigidBodyModel, SkinnedModel
from body_models.catalog import PUBLIC_MODULES
from body_models.constants import Joint
from body_models.registry import create_model, list_models

# Keep the concise ``body_models.smpl`` import surface while source code lives
# in semantic families such as ``body_models.bodies`` and ``body_models.robots``.
for _name, _target in PUBLIC_MODULES.items():
    _module = import_module(f".{_target}", __name__)
    globals()[_name] = _module
    sys.modules[f"{__name__}.{_name}"] = _module


__all__ = [
    *PUBLIC_MODULES,
    "Joint",
    "RigidBodyModel",
    "SkinnedModel",
    "create_model",
    "list_models",
]
