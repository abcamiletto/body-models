"""Public API for multi-runtime parametric and articulated body models."""

from body_models._base import (
    ArticulatedModel,
    CorrectiveBasis,
    DenseCorrectiveBasis,
    LinearIdentity,
    ParameterRole,
    ParameterSpec,
    PointRegressor,
    RigidBodyModel,
    SkinnedModel,
    SkinningIdentity,
    SkinningPose,
    SkinningSpec,
    SparseCorrectiveBasis,
)
from body_models._common.sparse import SparseMatrix
from body_models._constants import Joint
from body_models._registry import create_model, list_models
from body_models._rotations import RotationType
from body_models._runtime import (
    ArrayRuntime,
    JaxRuntime,
    KernelBackend,
    NumpyRuntime,
    RuntimeName,
    TorchRuntime,
)

for _public in (
    ArticulatedModel,
    ArrayRuntime,
    DenseCorrectiveBasis,
    JaxRuntime,
    Joint,
    LinearIdentity,
    NumpyRuntime,
    ParameterSpec,
    PointRegressor,
    RigidBodyModel,
    SkinnedModel,
    SkinningIdentity,
    SkinningPose,
    SkinningSpec,
    SparseCorrectiveBasis,
    SparseMatrix,
    TorchRuntime,
    create_model,
    list_models,
):
    _public.__module__ = __name__
del _public

__all__ = [
    "ArrayRuntime",
    "ArticulatedModel",
    "CorrectiveBasis",
    "DenseCorrectiveBasis",
    "JaxRuntime",
    "Joint",
    "KernelBackend",
    "LinearIdentity",
    "NumpyRuntime",
    "ParameterRole",
    "ParameterSpec",
    "PointRegressor",
    "RigidBodyModel",
    "RotationType",
    "RuntimeName",
    "SkinnedModel",
    "SkinningIdentity",
    "SkinningPose",
    "SkinningSpec",
    "SparseCorrectiveBasis",
    "SparseMatrix",
    "TorchRuntime",
    "create_model",
    "list_models",
]
