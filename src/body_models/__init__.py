"""Public API for multi-runtime parametric and articulated body models."""

from body_models._base import ParameterRole, ParameterSpec, RigidBodyModel, SkinnedModel
from body_models._constants import Joint
from body_models._registry import create_model, list_models
from body_models._rotations import RotationType
from body_models._runtime import (
    ArrayRuntime,
    Backend,
    JaxRuntime,
    NumpyRuntime,
    RuntimeLike,
    TorchRuntime,
)

for _public in (
    ArrayRuntime,
    JaxRuntime,
    Joint,
    NumpyRuntime,
    ParameterSpec,
    RigidBodyModel,
    SkinnedModel,
    TorchRuntime,
    create_model,
    list_models,
):
    _public.__module__ = __name__
del _public

__all__ = [
    "ArrayRuntime",
    "Backend",
    "JaxRuntime",
    "Joint",
    "NumpyRuntime",
    "ParameterRole",
    "ParameterSpec",
    "RigidBodyModel",
    "RotationType",
    "RuntimeLike",
    "SkinnedModel",
    "TorchRuntime",
    "create_model",
    "list_models",
]
