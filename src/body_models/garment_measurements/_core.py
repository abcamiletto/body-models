"""Backend-independent GarmentMeasurements identity and pose preparation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

from jaxtyping import Float
from nanomanifold import SE3, SO3

from body_models import _common as common
from body_models._rotations import RotationType, rotation_ndim
from body_models._runtime import ArrayRuntime

Array = Any


class GarmentMeasurementsIdentity(TypedDict):
    """Shape-dependent GarmentMeasurements state."""

    rest_vertices: Float[Array, "*batch V 3"]
    bind_skeleton: Float[Array, "*batch J 7"]
    local_bind_translations: Float[Array, "*batch J 3"]


def prepare_pose(
    runtime: ArrayRuntime,
    bind_quats: Float[Array, "J 4"],
    tree: common.KinematicTree,
    pose: Float[Array, "*batch J N"] | Float[Array, "*batch J 3 3"],
    rotation_type: RotationType,
    *,
    bind_skeleton: Float[Array, "*batch J 7"],
    local_bind_translations: Float[Array, "*batch J 3"],
) -> common.deformation.SkinningPose:
    """Prepare posed skeleton and bind-relative skinning transforms."""
    xp = runtime.xp
    num_rot_dims = rotation_ndim(rotation_type)
    batch_shape = tuple(pose.shape[: -(num_rot_dims + 1)])
    bind_skeleton = xp.broadcast_to(bind_skeleton, (*batch_shape, *bind_skeleton.shape[-2:]))
    local_bind_translations = xp.broadcast_to(
        local_bind_translations,
        (*batch_shape, *local_bind_translations.shape[-2:]),
    )
    skeleton = _forward_skeleton(
        bind_quats=bind_quats,
        local_bind_translations=local_bind_translations,
        tree=tree,
        pose=pose,
        rotation_type=rotation_type,
        runtime=runtime,
    )
    bind_matrices = SE3.to_matrix(bind_skeleton, xp=xp)
    skinning_transforms = skeleton @ common.invert_rigid_transforms(bind_matrices, xp=xp)
    return {
        "skeleton_transforms": skeleton,
        "skinning_transforms": skinning_transforms,
    }


def prepare_skeleton(
    runtime: ArrayRuntime,
    bind_quats: Float[Array, "J 4"],
    tree: common.KinematicTree,
    pose: Float[Array, "*batch J N"] | Float[Array, "*batch J 3 3"],
    rotation_type: RotationType,
    *,
    local_bind_translations: Float[Array, "*batch J 3"],
    joint_indices: Sequence[int] | None = None,
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed GarmentMeasurements joint transforms."""
    xp = runtime.xp
    batch_shape = pose.shape[: -(rotation_ndim(rotation_type) + 1)]
    local_bind_translations = xp.broadcast_to(
        local_bind_translations,
        (*batch_shape, *local_bind_translations.shape[-2:]),
    )
    return _forward_skeleton(
        bind_quats=bind_quats,
        local_bind_translations=local_bind_translations,
        tree=tree,
        pose=pose,
        rotation_type=rotation_type,
        runtime=runtime,
        joint_indices=joint_indices,
    )


def prepare_identity(
    *,
    xp: Any,
    mean_vertices: Float[Array, "V 3"],
    components: Float[Array, "V 3 C"],
    eigenvalues: Float[Array, "C"],
    bind_quats: Float[Array, "J 4"],
    mvc_weights: Float[Array, "V J"],
    kinematic_tree: common.KinematicTree,
    shape: Float[Array, "*batch C"],
) -> GarmentMeasurementsIdentity:
    """Prepare shape-dependent surface and bind skeleton."""
    if shape.ndim < 1 or shape.shape[-1] != eigenvalues.shape[0]:
        raise ValueError(f"shape must have shape [..., {eigenvalues.shape[0]}]")
    scaled_shape = shape * xp.sqrt(eigenvalues)
    rest_vertices = mean_vertices + xp.einsum("...c,vdc->...vd", scaled_shape, components)
    joint_positions = xp.einsum("vj,...vd->...jd", mvc_weights, rest_vertices)
    bind_quats = xp.broadcast_to(bind_quats, (*rest_vertices.shape[:-2], *bind_quats.shape))
    bind_global_quats = _propagate_quats(bind_quats, kinematic_tree.fronts, xp=xp)
    local_translations = _local_translations_from_positions(
        joint_positions,
        bind_global_quats,
        kinematic_tree.fronts,
        xp=xp,
    )
    return {
        "rest_vertices": rest_vertices,
        "bind_skeleton": SE3.from_rt(bind_global_quats, joint_positions, xp=xp),
        "local_bind_translations": local_translations,
    }


def _forward_skeleton(
    runtime: ArrayRuntime,
    *,
    bind_quats: Float[Array, "J 4"],
    local_bind_translations: Float[Array, "*batch J 3"],
    tree: common.KinematicTree,
    pose: Float[Array, "*batch J N"] | Float[Array, "*batch J 3 3"],
    rotation_type: RotationType,
    joint_indices: Sequence[int] | None = None,
) -> Float[Array, "*batch J 4 4"]:
    xp = runtime.xp
    batch_shape = local_bind_translations.shape[:-2]
    bind_quats = xp.broadcast_to(bind_quats, (*batch_shape, *bind_quats.shape))
    selection = None
    if joint_indices is not None:
        selection = tree.select(joint_indices)
        cover_indices = xp.asarray(selection.cover_indices, dtype=xp.int32)
        if rotation_ndim(rotation_type) > 1:
            pose = pose[..., cover_indices, :, :]
        else:
            pose = pose[..., cover_indices, :]
        bind_quats = bind_quats[..., cover_indices, :]
        local_bind_translations = local_bind_translations[..., cover_indices, :]
        tree = selection.tree
    pose_quats = SO3.convert(pose, src=rotation_type, dst="quat", xp=xp)
    posed_quats = SO3.multiply(bind_quats, pose_quats, xp=xp)
    local_rotations = SO3.convert(posed_quats, src="quat", dst="rotmat", xp=xp)
    local_transforms = common.affine_transforms(local_rotations, local_bind_translations, xp=xp)
    skeleton = runtime._compose_kinematic_tree(local_transforms, tree)
    if selection is None:
        return skeleton
    output_indices = xp.asarray(selection.output_indices, dtype=xp.int32)
    return skeleton[..., output_indices, :, :]


def _local_translations_from_positions(
    positions: Float[Array, "*batch J 3"],
    bind_global_quats: Float[Array, "*batch J 4"],
    fronts: Sequence[common.Front],
    *,
    xp: Any,
) -> Float[Array, "*batch J 3"]:
    translations = xp.zeros_like(positions)
    for joints, parents in fronts:
        if parents[0] < 0:
            front = positions[..., joints, :]
        else:
            offsets = positions[..., joints, :] - positions[..., parents, :]
            parent_inverse = SO3.inverse(bind_global_quats[..., parents, :], xp=xp)
            front = SO3.rotate_points(parent_inverse, offsets[..., None, :], xp=xp).squeeze(-2)
        translations = common.at_set(
            translations,
            (..., joints, slice(None)),
            front,
            copy=False,
            xp=xp,
        )
    return translations


def _propagate_quats(
    quaternions: Float[Array, "*batch J 4"],
    fronts: Sequence[common.Front],
    *,
    xp: Any,
) -> Float[Array, "*batch J 4"]:
    global_quaternions = xp.zeros_like(quaternions)
    for joints, parents in fronts:
        if parents[0] < 0:
            front = quaternions[..., joints, :]
        else:
            front = SO3.multiply(
                global_quaternions[..., parents, :],
                quaternions[..., joints, :],
                xp=xp,
            )
        global_quaternions = common.at_set(
            global_quaternions,
            (..., joints, slice(None)),
            front,
            copy=False,
            xp=xp,
        )
    return global_quaternions


__all__ = [
    "GarmentMeasurementsIdentity",
    "prepare_identity",
    "prepare_pose",
]
