"""Backend-independent FLAME pose and identity preparation."""

from collections.abc import Sequence
from typing import Any

from jaxtyping import Float

from body_models import _common as common
from body_models import _linear_blendshape as linear
from body_models._common import deformation
from body_models._rotations import RotationType
from body_models._runtime import ArrayRuntime

Array = Any

FlameSkeletonIdentity = deformation.SkeletonIdentity

prepare_identity = linear.prepare_shape_expression_identity
prepare_skeleton_identity = linear.prepare_shape_expression_skeleton_identity


def _pose_matrices(
    runtime: ArrayRuntime,
    head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
    head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    selection: common.JointSelection | None = None,
) -> Float[Array, "*batch 5 3 3"]:
    return linear.assemble_pose_matrices(
        runtime,
        [linear.PoseBlock(head_pose, rotation_type)],
        head_rotation,
        rotation_type,
        selection,
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
        runtime,
        head_pose,
        head_rotation,
        rotation_type,
    )
    return linear.prepare_pose(
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
    selection = None if joint_indices is None else tree.select(joint_indices)
    pose_matrices = _pose_matrices(
        runtime,
        head_pose,
        head_rotation,
        rotation_type,
        selection,
    )
    return linear.forward_skeleton(
        runtime,
        tree,
        pose_matrices,
        local_joint_offsets,
        selection=selection,
    )


__all__ = ["prepare_identity", "prepare_pose"]
