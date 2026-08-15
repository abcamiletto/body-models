"""Private numerical operations shared across model programs."""

from body_models._common import deformation, sparse
from body_models._common.kinematics import (
    Front,
    JointSelection,
    KinematicTree,
    affine_transforms,
    compose_kinematic_fronts,
    compute_sparse_skin_weights,
    invert_rigid_transforms,
    local_joint_offsets,
    rotation_between_vectors,
)
from body_models._common.ops import Array, at_set, eye_as, take_along_axis, zeros_as
from body_models._common.simplify_mesh import simplify_mesh

__all__ = [
    "Array",
    "Front",
    "JointSelection",
    "KinematicTree",
    "affine_transforms",
    "at_set",
    "compose_kinematic_fronts",
    "compute_sparse_skin_weights",
    "deformation",
    "eye_as",
    "invert_rigid_transforms",
    "local_joint_offsets",
    "rotation_between_vectors",
    "simplify_mesh",
    "sparse",
    "take_along_axis",
    "zeros_as",
]
