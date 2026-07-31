"""Public API for multi-runtime parametric and articulated body models."""

from body_models._base import (
    ArticulatedModel,
    LinearIdentity,
    ParameterRole,
    ParameterSpec,
    RigidBodyModel,
    SkinnedModel,
    SkinningIdentity,
    SkinningPayload,
    SkinningPose,
)
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
    ArticulatedModel,
    ArrayRuntime,
    JaxRuntime,
    Joint,
    LinearIdentity,
    NumpyRuntime,
    ParameterSpec,
    RigidBodyModel,
    SkinnedModel,
    SkinningIdentity,
    SkinningPayload,
    SkinningPose,
    TorchRuntime,
    create_model,
    list_models,
):
    _public.__module__ = __name__
del _public

__all__ = [
    "ArrayRuntime",
    "ArticulatedModel",
    "Backend",
    "JaxRuntime",
    "Joint",
    "LinearIdentity",
    "NumpyRuntime",
    "ParameterRole",
    "ParameterSpec",
    "RigidBodyModel",
    "RotationType",
    "RuntimeLike",
    "SkinnedModel",
    "SkinningIdentity",
    "SkinningPayload",
    "SkinningPose",
    "TorchRuntime",
    "create_model",
    "list_models",
]
