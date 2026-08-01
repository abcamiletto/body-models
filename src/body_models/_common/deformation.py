"""Backend-agnostic linear deformation primitives."""

from dataclasses import dataclass
from typing import Any, NotRequired, TypeAlias, TypedDict

from jaxtyping import Float, Int

from body_models._common import kinematics, ops, sparse

Array = Any


class SkeletonIdentity(TypedDict):
    """Identity-dependent joint state shared by linear body models."""

    rest_joints: Float[Array, "*batch J 3"]
    local_joint_offsets: Float[Array, "*batch J 3"]


class SkinningIdentity(TypedDict):
    """Identity-dependent rest surface consumed by skinning."""

    rest_vertices: Float[Array, "*batch V 3"]


class LinearIdentity(SkeletonIdentity, SkinningIdentity):
    """Identity-dependent joints and vertices from linear bases."""


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


def prepare_linear_identity(
    *,
    vertex_template: Float[Array, "V 3"],
    vertex_directions: Float[Array, "V 3 C"],
    joint_template: Float[Array, "J 3"],
    joint_directions: Float[Array, "J 3 C"],
    parents: list[int],
    coefficients: Float[Array, "*batch C"],
    xp: Any,
) -> LinearIdentity:
    """Prepare joints and vertices controlled by the same linear coefficients."""
    skeleton = prepare_linear_skeleton(
        joint_template=joint_template,
        joint_directions=joint_directions,
        parents=parents,
        coefficients=coefficients,
        xp=xp,
    )
    return {
        "rest_joints": skeleton["rest_joints"],
        "local_joint_offsets": skeleton["local_joint_offsets"],
        "rest_vertices": blend_shapes(
            vertex_template,
            vertex_directions,
            coefficients,
            xp=xp,
        ),
    }


def prepare_linear_skeleton(
    *,
    joint_template: Float[Array, "J 3"],
    joint_directions: Float[Array, "J 3 C"],
    parents: list[int],
    coefficients: Float[Array, "*batch C"],
    xp: Any,
) -> SkeletonIdentity:
    """Prepare joints controlled by a linear coefficient basis."""
    if coefficients.ndim < 1 or coefficients.shape[-1] < 1:
        raise ValueError("coefficients must have shape [..., C] with C >= 1")
    rest_joints = blend_shapes(
        joint_template,
        joint_directions,
        coefficients,
        xp=xp,
    )
    return {
        "rest_joints": rest_joints,
        "local_joint_offsets": kinematics.local_joint_offsets(
            rest_joints,
            parents,
            xp=xp,
        ),
    }


def blend_shapes(
    mean: Float[Array, "V D"],
    directions: Float[Array, "V D C"],
    coefficients: Float[Array, "*batch C"],
    *,
    xp: Any,
) -> Float[Array, "*batch V D"]:
    """Apply a linear blend-shape basis stored along its final axis."""
    if directions.shape[-1] != coefficients.shape[-1]:
        raise ValueError("directions and coefficients must have the same component count")
    return mean + xp.einsum("...c,vdc->...vd", coefficients, directions)


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
    "SkeletonIdentity",
    "SkinningIdentity",
    "SkinningPose",
    "SkinningSpec",
    "SparseCorrectiveBasis",
    "blend_shapes",
    "pose_coefficients",
    "prepare_linear_identity",
    "prepare_linear_skeleton",
]
