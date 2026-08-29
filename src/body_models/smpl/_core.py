"""SMPL deformation computations."""

from collections.abc import Sequence
from typing import Any

from jaxtyping import Float

from body_models import _common as common
from body_models import _linear_blendshape as linear
from body_models._common import deformation
from body_models._rotations import RotationType
from body_models._runtime import ArrayRuntime

Array = Any
SmplSkeletonIdentity = deformation.SkeletonIdentity

prepare_identity = linear.prepare_shape_identity
prepare_skeleton_identity = linear.prepare_shape_skeleton_identity


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
    pose_matrices = linear.assemble_pose_matrices(
        runtime,
        [linear.PoseBlock(body_pose, rotation_type)],
        pelvis_rotation,
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
    body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    joint_indices: Sequence[int] | None = None,
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed SMPL joint transforms."""
    selection = None if joint_indices is None else tree.select(joint_indices)
    pose_matrices = linear.assemble_pose_matrices(
        runtime,
        [linear.PoseBlock(body_pose, rotation_type)],
        pelvis_rotation,
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
