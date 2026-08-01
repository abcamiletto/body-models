"""Backend-independent FLAME pose and identity preparation."""

from typing import Any

from jaxtyping import Float

from body_models import _smpl_family as family
from body_models._common import deformation
from body_models._rotations import RotationType

Array = Any
Front = tuple[list[int], list[int]]

FlameSkeletonIdentity = deformation.SkeletonIdentity

prepare_identity = family.prepare_shape_expression_identity
prepare_skeleton_identity = family.prepare_shape_expression_skeleton_identity


def _pose_matrices(
    head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
    head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    xp: Any,
) -> Float[Array, "*batch 5 3 3"]:
    return family.assemble_pose_matrices(
        [(head_pose, rotation_type)],
        head_rotation,
        rotation_type,
        xp=xp,
    )


def prepare_pose(
    kinematic_fronts: list[Front],
    head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
    head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    rest_joints: Float[Array, "*identity_batch J 3"],
    xp: Any,
) -> deformation.SkinningPose:
    """Prepare FLAME transforms and pose-corrective coefficients."""
    pose_matrices = _pose_matrices(
        head_pose,
        head_rotation,
        rotation_type,
        xp=xp,
    )
    return family.prepare_pose(
        pose_matrices,
        kinematic_fronts,
        local_joint_offsets=local_joint_offsets,
        rest_joints=rest_joints,
        xp=xp,
    )


def prepare_skeleton(
    kinematic_fronts: list[Front],
    head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
    head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    xp: Any,
) -> Float[Array, "*batch 5 4 4"]:
    """Prepare only posed FLAME joint transforms."""
    pose_matrices = _pose_matrices(
        head_pose,
        head_rotation,
        rotation_type,
        xp=xp,
    )
    return family.forward_skeleton(
        pose_matrices,
        local_joint_offsets,
        kinematic_fronts,
        xp=xp,
    )


__all__ = ["prepare_identity", "prepare_pose"]
