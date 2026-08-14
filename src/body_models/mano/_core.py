"""Backend-independent MANO pose and identity preparation."""

from collections.abc import Sequence
from typing import Any

from jaxtyping import Float

from body_models import _common as common
from body_models import _smpl_family as family
from body_models._common import deformation
from body_models._rotations import RotationType
from body_models._runtime import ArrayRuntime

Array = Any

ManoSkeletonIdentity = deformation.SkeletonIdentity

prepare_identity = family.prepare_shape_identity
prepare_skeleton_identity = family.prepare_shape_skeleton_identity


def _pose_matrices(
    hand_mean: Float[Array, "45"],
    hand_pose: Float[Array, "*batch 15 N"] | Float[Array, "*batch 15 3 3"],
    wrist_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    xp: Any,
    positions: Sequence[int] | None = None,
) -> Float[Array, "*batch 16 3 3"]:
    hand_pose = family.take_pose_joints(hand_pose, positions, rotation_type, xp=xp)
    if positions is not None:
        hand_mean = hand_mean.reshape(-1, 3)[xp.asarray(positions, dtype=xp.int32)]
    hand_axis_angle = family.add_axis_angle_mean(
        hand_pose,
        hand_mean,
        rotation_type,
        xp=xp,
    )
    return family.assemble_pose_matrices(
        [(hand_axis_angle, "axis_angle")],
        wrist_rotation,
        rotation_type,
        xp=xp,
    )


def prepare_pose(
    runtime: ArrayRuntime,
    tree: common.KinematicTree,
    hand_mean: Float[Array, "45"],
    hand_pose: Float[Array, "*batch 15 N"] | Float[Array, "*batch 15 3 3"],
    wrist_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    rest_joints: Float[Array, "*identity_batch J 3"],
) -> deformation.SkinningPose:
    """Prepare MANO transforms and pose-corrective coefficients."""
    pose_matrices = _pose_matrices(
        hand_mean,
        hand_pose,
        wrist_rotation,
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
    hand_mean: Float[Array, "45"],
    hand_pose: Float[Array, "*batch 15 N"] | Float[Array, "*batch 15 3 3"],
    wrist_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    joint_indices: Sequence[int] | None = None,
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed MANO joint transforms."""
    subtree = positions = None
    if joint_indices is not None:
        subtree, positions = family.pose_joint_positions(tree, joint_indices)
    pose_matrices = _pose_matrices(
        hand_mean,
        hand_pose,
        wrist_rotation,
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
