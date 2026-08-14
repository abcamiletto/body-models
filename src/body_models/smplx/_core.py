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
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    xp: Any,
    positions: Sequence[int] | None = None,
) -> Float[Array, "*batch 55 3 3"]:
    body_positions = head_positions = hand_positions = None
    if positions is not None:
        body_positions = tuple(position for position in positions if position < 21)
        head_positions = tuple(position - 21 for position in positions if 21 <= position < 24)
        hand_positions = tuple(position - 24 for position in positions if position >= 24)
    body_pose = family.take_pose_joints(body_pose, body_positions, rotation_type, xp=xp)
    head_pose = family.take_pose_joints(head_pose, head_positions, rotation_type, xp=xp)
    hand_pose = family.take_pose_joints(hand_pose, hand_positions, rotation_type, xp=xp)
    if hand_positions is not None:
        hand_mean = hand_mean.reshape(-1, 3)[xp.asarray(hand_positions, dtype=xp.int32)]
    hand_axis_angle = family.add_axis_angle_mean(
        hand_pose,
        hand_mean,
        rotation_type,
        xp=xp,
    )
    return family.assemble_pose_matrices(
        [
            (body_pose, rotation_type),
            (head_pose, rotation_type),
            (hand_axis_angle, "axis_angle"),
        ],
        pelvis_rotation,
        rotation_type,
        xp=xp,
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
        hand_mean,
        body_pose,
        head_pose,
        hand_pose,
        pelvis_rotation,
        rotation_type,
        xp=runtime.xp,
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
    subtree = positions = None
    if joint_indices is not None:
        subtree, positions = family.pose_joint_positions(tree, joint_indices)
    pose_matrices = _pose_matrices(
        hand_mean,
        body_pose,
        head_pose,
        hand_pose,
        pelvis_rotation,
        rotation_type,
        xp=runtime.xp,
        positions=positions,
    )
    return family.forward_skeleton(
        runtime,
        tree,
        pose_matrices,
        local_joint_offsets,
        subtree=subtree,
    )


__all__ = ["prepare_identity", "prepare_pose"]
