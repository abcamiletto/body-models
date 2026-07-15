"""Multi-runtime parametric and articulated body models."""

from importlib import import_module

from body_models.base import RigidBodyModel, SkinnedModel
from body_models.catalog import PUBLIC_MODULES
from body_models.constants import Joint
from body_models.registry import create_model, list_models

# Import the lightweight public packages without loading any array backend.
for _name, _target in PUBLIC_MODULES.items():
    _module = import_module(f".{_target}", __name__)
    globals()[_name] = _module


__all__ = [
    *PUBLIC_MODULES,
    "Joint",
    "RigidBodyModel",
    "SkinnedModel",
    "create_model",
    "list_models",
]
