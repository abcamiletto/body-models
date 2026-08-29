"""Backend-independent GNM Head identity and pose preparation."""

from collections.abc import Sequence
from typing import Any

from jaxtyping import Float

from body_models import _common as common
from body_models import _linear_blendshape as linear
from body_models._common import deformation
from body_models._rotations import RotationType
from body_models._runtime import ArrayRuntime

Array = Any


def prepare_identity(
    *,
    xp: Any,
    v_template: Float[Array, "V 3"],
    shapedirs: Float[Array, "V 3 S"],
    exprdirs: Float[Array, "V 3 E"],
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: Sequence[int],
    shape: Float[Array, "*batch S"],
    expression: Float[Array, "*batch E"],
) -> deformation.LinearIdentity:
    """Prepare GNM's identity-dependent joints and rest surface."""
    skeleton = deformation.prepare_linear_skeleton(
        joint_template=j_template,
        joint_directions=j_shapedirs,
        parents=parents,
        coefficients=shape,
        xp=xp,
    )
    rest_vertices = deformation.blend_shapes(v_template, shapedirs, shape, xp=xp)
    rest_vertices = rest_vertices + xp.einsum("...e,vde->...vd", expression, exprdirs)
    return {
        "rest_joints": skeleton["rest_joints"],
        "local_joint_offsets": skeleton["local_joint_offsets"],
        "rest_vertices": rest_vertices,
    }


def prepare_skeleton_identity(
    *,
    xp: Any,
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: Sequence[int],
    shape: Float[Array, "*batch S"],
) -> deformation.SkeletonIdentity:
    """Prepare GNM's identity-dependent joints without the surface."""
    return deformation.prepare_linear_skeleton(
        joint_template=j_template,
        joint_directions=j_shapedirs,
        parents=parents,
        coefficients=shape,
        xp=xp,
    )


def pose_matrices(
    runtime: ArrayRuntime,
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    selection: common.JointSelection | None = None,
) -> Float[Array, "*batch 4 3 3"]:
    """Convert GNM's ordered joint controls to rotation matrices."""
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
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch 4 3"],
    rest_joints: Float[Array, "*identity_batch 4 3"],
) -> deformation.SkinningPose:
    """Prepare GNM skinning transforms and its root-inclusive correctives."""
    rotations = pose_matrices(runtime, head_pose, head_rotation, rotation_type)
    pose = linear.prepare_pose(
        runtime,
        tree,
        rotations,
        local_joint_offsets=local_joint_offsets,
        rest_joints=rest_joints,
    )
    xp = runtime.xp
    batch_shape = rotations.shape[:-3]
    identity = common.ops.eye_as(rotations, batch_dims=(*batch_shape, 1), xp=xp)
    pose["pose_coefficients"] = (rotations - identity).reshape(*batch_shape, -1)
    return pose


def prepare_skeleton(
    runtime: ArrayRuntime,
    tree: common.KinematicTree,
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch 4 3"],
    joint_indices: Sequence[int] | None = None,
) -> Float[Array, "*batch selected 4 4"]:
    """Prepare GNM world-space joint transforms."""
    selection = None if joint_indices is None else tree.select(joint_indices)
    rotations = pose_matrices(runtime, head_pose, head_rotation, rotation_type, selection)
    return linear.forward_skeleton(runtime, tree, rotations, local_joint_offsets, selection=selection)


__all__ = ["prepare_identity", "prepare_pose", "prepare_skeleton", "prepare_skeleton_identity"]
