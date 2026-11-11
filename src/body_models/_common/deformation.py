"""Backend-agnostic linear deformation primitives."""

from dataclasses import dataclass
from typing import Any, NotRequired, TypeAlias, TypedDict

from jaxtyping import Float, Int

from body_models._common import ops, sparse

Array = Any


class LinearSkeletonIdentity(TypedDict):
    """Identity coefficients and the rest joints they produce."""

    coefficients: tuple[Float[Array, "*batch C"], ...]
    rest_joints: Float[Array, "*batch J 3"]
    local_joint_offsets: Float[Array, "*batch J 3"]


class SkinningIdentity(TypedDict):
    """Identity-dependent rest surface consumed by skinning."""

    rest_vertices: Float[Array, "*batch V 3"]


class LinearIdentity(LinearSkeletonIdentity, SkinningIdentity):
    """Identity coefficients and the rest joints and vertices they produce."""


class SkinningPose(TypedDict):
    """Pose-dependent state shared by linear blend skinning models."""

    skeleton_transforms: Float[Array, "*batch J 4 4"]
    skinning_transforms: Float[Array, "*batch J 4 4"]
    pose_coefficients: NotRequired[Float[Array, "*batch C"]]


@dataclass(frozen=True)
class DenseCorrectiveBasis:
    """Dense corrective basis stored as ``[coefficients, vertex coordinates]``."""

    values: Float[Array, "C V*3"]

    @property
    def coefficient_dim(self) -> int:
        return self.values.shape[0]

    @property
    def num_vertices(self) -> int:
        return self.values.shape[1] // 3

    def apply(self, coefficients: Float[Array, "*batch C"]) -> Float[Array, "*batch V 3"]:
        return (coefficients @ self.values).reshape(*coefficients.shape[:-1], self.num_vertices, 3)


@dataclass(frozen=True)
class SparseCorrectiveBasis:
    """Sparse corrective basis backed by the active array runtime."""

    linear: sparse.SparseLinear

    @property
    def coefficient_dim(self) -> int:
        return self.linear.shape[0]

    @property
    def num_vertices(self) -> int:
        return self.linear.shape[1] // 3

    def apply(self, coefficients: Float[Array, "*batch C"]) -> Float[Array, "*batch V 3"]:
        offsets = self.linear(coefficients)
        return offsets.reshape(*coefficients.shape[:-1], self.num_vertices, 3)

    def to_coo(self) -> sparse.SparseMatrix:
        """Return the basis in coordinate format without densifying it."""
        return self.linear.to_coo()


CorrectiveBasis: TypeAlias = DenseCorrectiveBasis | SparseCorrectiveBasis


@dataclass(frozen=True)
class SkinningSpec:
    """Model-static data consumed by linear blend skinning renderers."""

    triangles: Int[Array, "F 3"]
    skinning_weights: Float[Array, "V J"]
    corrective_basis: CorrectiveBasis | None = None


def blend_shapes(
    mean: Float[Array, "*batch V D"],
    directions: Float[Array, "V D C"],
    coefficients: Float[Array, "*batch C"],
    *,
    xp: Any,
) -> Float[Array, "*batch V D"]:
    """Apply a linear blend-shape basis stored along its final axis."""
    if directions.shape[-1] != coefficients.shape[-1]:
        raise ValueError("directions and coefficients must have the same component count")
    flat_directions = directions.reshape(directions.shape[0] * directions.shape[1], directions.shape[-1])
    offsets = coefficients @ flat_directions.mT
    return mean + offsets.reshape(*coefficients.shape[:-1], *directions.shape[:-1])


def pose_coefficients(
    rotations: Float[Array, "*batch J 3 3"],
    *,
    xp: Any,
) -> Float[Array, "*batch C"]:
    """Flatten root-excluded joint rotation deviations."""
    batch_shape = rotations.shape[:-3]
    identity = ops.eye_as(rotations, batch_dims=(*batch_shape, 1), xp=xp)
    return (rotations[..., 1:, :, :] - identity).reshape(*batch_shape, -1)


__all__ = [
    "CorrectiveBasis",
    "DenseCorrectiveBasis",
    "LinearIdentity",
    "LinearSkeletonIdentity",
    "SkinningIdentity",
    "SkinningPose",
    "SkinningSpec",
    "SparseCorrectiveBasis",
    "blend_shapes",
    "pose_coefficients",
]
