"""SMPL deformation computations."""

from collections.abc import Sequence
from typing import Any

from jaxtyping import Float

from body_models import _common as common
from body_models import _smpl_family as family
from body_models._common import deformation
from body_models._rotations import RotationType
from body_models._runtime import ArrayRuntime

Array = Any
SmplSkeletonIdentity = deformation.SkeletonIdentity

prepare_identity = family.prepare_shape_identity
prepare_skeleton_identity = family.prepare_shape_skeleton_identity


def prepare_pose(
    runtime: ArrayRuntime,
    tree: common.KinematicTree,
    body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    rest_joints: Float[Array, "*identity_batch J 3"],
) -> deformation.SkinningPose:
    """Prepare SMPL transforms and pose-corrective coefficients."""
    pose_matrices = family.assemble_pose_matrices(
        [(body_pose, rotation_type)],
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
    body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    joint_indices: Sequence[int] | None = None,
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed SMPL joint transforms."""
    subtree = positions = None
    if joint_indices is not None:
        subtree, positions = family.pose_joint_positions(tree, joint_indices)
    body_pose = family.take_pose_joints(body_pose, positions, rotation_type, xp=runtime.xp)
    pose_matrices = family.assemble_pose_matrices(
        [(body_pose, rotation_type)],
        pelvis_rotation,
        rotation_type,
        xp=runtime.xp,
    )
    return family.forward_skeleton(
        runtime,
        tree,
        pose_matrices,
        local_joint_offsets,
        subtree=subtree,
    )


__all__ = ["prepare_identity", "prepare_pose"]
