"""Backend-independent SMPL-X pose and identity preparation."""

from collections.abc import Sequence
from typing import Any

from jaxtyping import Float

from body_models import _common as common
from body_models import _smpl_family as family
from body_models._common import deformation
from body_models._rotations import RotationType
from body_models._runtime import ArrayRuntime

Array = Any

SmplxSkeletonIdentity = deformation.SkeletonIdentity

prepare_identity = family.prepare_shape_expression_identity
prepare_skeleton_identity = family.prepare_shape_expression_skeleton_identity


def _pose_matrices(
    runtime: ArrayRuntime,
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    selection: common.JointSelection | None = None,
) -> Float[Array, "*batch 55 3 3"]:
    return family.assemble_pose_matrices(
        runtime,
        [
            family.PoseBlock(body_pose, rotation_type),
            family.PoseBlock(head_pose, rotation_type),
            family.PoseBlock(hand_pose, rotation_type, axis_angle_mean=hand_mean),
        ],
        pelvis_rotation,
        rotation_type,
        selection,
    )


def prepare_pose(
    runtime: ArrayRuntime,
    tree: common.KinematicTree,
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    rest_joints: Float[Array, "*identity_batch J 3"],
) -> deformation.SkinningPose:
    """Prepare SMPL-X transforms and pose-corrective coefficients."""
    pose_matrices = _pose_matrices(
        runtime,
        hand_mean,
        body_pose,
        head_pose,
        hand_pose,
        pelvis_rotation,
        rotation_type,
    )
    return family.prepare_pose(
        runtime,
        tree,
        pose_matrices,
        local_joint_offsets=local_joint_offsets,
        rest_joints=rest_joints,
    )


def prepare_skeleton(
    runtime: ArrayRuntime,
    tree: common.KinematicTree,
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    joint_indices: Sequence[int] | None = None,
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed SMPL-X joint transforms."""
    selection = None if joint_indices is None else tree.select(joint_indices)
    pose_matrices = _pose_matrices(
        runtime,
        hand_mean,
        body_pose,
        head_pose,
        hand_pose,
        pelvis_rotation,
        rotation_type,
        selection,
    )
    return family.forward_skeleton(
        runtime,
        tree,
        pose_matrices,
        local_joint_offsets,
        selection=selection,
    )


__all__ = ["prepare_identity", "prepare_pose"]
