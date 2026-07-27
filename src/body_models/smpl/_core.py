"""SMPL deformation computations."""

from typing import Any

from jaxtyping import Float

from body_models import _smpl_family as family
from body_models._common import deformation
from body_models._rotations import RotationType

Array = Any
Front = tuple[list[int], list[int]]

SmplSkeletonIdentity = deformation.SkeletonIdentity
SmplIdentity = deformation.LinearIdentity
SmplPreparedPose = deformation.SkinningPose

prepare_identity = family.prepare_shape_identity
prepare_skeleton_identity = family.prepare_shape_skeleton_identity


def prepare_pose(
    posedirs: Float[Array, "P V*3"],
    kinematic_fronts: list[Front],
    body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    rest_joints: Float[Array, "*identity_batch J 3"],
    xp: Any,
) -> SmplPreparedPose:
    """Prepare SMPL transforms and pose-dependent vertex offsets."""
    pose_matrices = family.assemble_pose_matrices(
        [(body_pose, rotation_type)],
        pelvis_rotation,
        rotation_type,
        xp=xp,
    )
    return family.prepare_pose(
        pose_matrices,
        posedirs,
        kinematic_fronts,
        local_joint_offsets=local_joint_offsets,
        rest_joints=rest_joints,
        xp=xp,
    )


def prepare_skeleton(
    kinematic_fronts: list[Front],
    body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    xp: Any,
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed SMPL joint transforms."""
    pose_matrices = family.assemble_pose_matrices(
        [(body_pose, rotation_type)],
        pelvis_rotation,
        rotation_type,
        xp=xp,
    )
    return family.forward_skeleton(
        pose_matrices,
        local_joint_offsets,
        kinematic_fronts,
        xp=xp,
    )


__all__ = [
    "SmplIdentity",
    "SmplPreparedPose",
    "prepare_identity",
    "prepare_pose",
]
