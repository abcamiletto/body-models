"""Backend-independent SMPL-H pose and identity preparation."""

from typing import Any

from jaxtyping import Float

from body_models import _smpl_family as family
from body_models._common import deformation
from body_models._rotations import RotationType

Array = Any
Front = tuple[list[int], list[int]]

SmplhSkeletonIdentity = deformation.SkeletonIdentity

prepare_identity = family.prepare_shape_identity
prepare_skeleton_identity = family.prepare_shape_skeleton_identity


def _pose_matrices(
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    xp: Any,
) -> Float[Array, "*batch 52 3 3"]:
    hand_axis_angle = family.add_axis_angle_mean(
        hand_pose,
        hand_mean,
        rotation_type,
        xp=xp,
    )
    return family.assemble_pose_matrices(
        [(body_pose, rotation_type), (hand_axis_angle, "axis_angle")],
        pelvis_rotation,
        rotation_type,
        xp=xp,
    )


def prepare_pose(
    posedirs: Float[Array, "P V*3"],
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    rest_joints: Float[Array, "*identity_batch J 3"],
    xp: Any,
) -> deformation.SkinningPose:
    """Prepare SMPL-H transforms and pose-dependent vertex offsets."""
    pose_matrices = _pose_matrices(
        hand_mean,
        body_pose,
        hand_pose,
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
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    xp: Any,
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed SMPL-H joint transforms."""
    pose_matrices = _pose_matrices(
        hand_mean,
        body_pose,
        hand_pose,
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


__all__ = ["prepare_identity", "prepare_pose"]
