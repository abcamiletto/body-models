"""Backend-independent FLAME pose and identity preparation."""

from collections.abc import Sequence
from typing import Any

from jaxtyping import Float

from body_models import _common as common
from body_models import _smpl_family as family
from body_models._common import deformation
from body_models._rotations import RotationType
from body_models._runtime import ArrayRuntime

Array = Any

FlameSkeletonIdentity = deformation.SkeletonIdentity

prepare_identity = family.prepare_shape_expression_identity
prepare_skeleton_identity = family.prepare_shape_expression_skeleton_identity


def _pose_matrices(
    head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
    head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    xp: Any,
    positions: Sequence[int] | None = None,
) -> Float[Array, "*batch 5 3 3"]:
    head_pose = family.take_pose_joints(head_pose, positions, rotation_type, xp=xp)
    return family.assemble_pose_matrices(
        [(head_pose, rotation_type)],
        head_rotation,
        rotation_type,
        xp=xp,
    )


def prepare_pose(
    runtime: ArrayRuntime,
    tree: common.KinematicTree,
    head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
    head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    rest_joints: Float[Array, "*identity_batch J 3"],
) -> deformation.SkinningPose:
    """Prepare FLAME transforms and pose-corrective coefficients."""
    pose_matrices = _pose_matrices(
        head_pose,
        head_rotation,
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
    head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
    head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    joint_indices: Sequence[int] | None = None,
) -> Float[Array, "*batch 5 4 4"]:
    """Prepare only posed FLAME joint transforms."""
    subtree = positions = None
    if joint_indices is not None:
        subtree, positions = family.pose_joint_positions(tree, joint_indices)
    pose_matrices = _pose_matrices(
        head_pose,
        head_rotation,
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
