"""SOMA identity, rigging, and pose mathematics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import _common as common
from body_models._common import skinning
from body_models._rotations import RotationType
from body_models._runtime import ArrayRuntime

if TYPE_CHECKING:
    from body_models.soma._schema import SomaWeights

Array = Any
BindPoseMode = Literal["fit", "fit_detached", "canonical"]


class SomaSkeletonIdentity(TypedDict):
    """Identity-dependent joint state needed to pose the SOMA skeleton."""

    local_joint_translations: Float[Array, "*batch Jf 3"]


class SomaIdentity(SomaSkeletonIdentity):
    """Complete identity-dependent SOMA mesh state."""

    rest_vertices: Float[Array, "*batch Va 3"]
    inverse_bind_transforms: Float[Array, "*batch Jf 4 4"]


def skinning_weights(data: SomaWeights) -> Float[Array, "Va Jf"]:
    return data.skin_weights_active[:, 1:]


def prepare_identity_from_rest_shape(
    runtime: ArrayRuntime,
    data: SomaWeights,
    *,
    rest_shape_full: Float[Array, "B Vf 3"],
    rest_shape_active: Float[Array, "B Va 3"],
    repose: bool = True,
    bind_pose: BindPoseMode = "fit",
) -> SomaIdentity:
    xp = runtime.xp
    bind_shape, world_bind_pose = _prepare_bind_state(
        data,
        rest_shape_full=rest_shape_full,
        rest_shape_active=rest_shape_active,
        runtime=runtime,
        repose=repose,
        bind_pose=bind_pose,
    )
    return _prepare_identity_state(xp, bind_shape, world_bind_pose, data.kinematics.kinematic_tree.parents)


def prepare_skeleton_identity_from_rest_shape(
    runtime: ArrayRuntime,
    data: SomaWeights,
    *,
    rest_shape_full: Float[Array, "B Vf 3"],
    rest_shape_active: Float[Array, "B Va 3"],
    repose: bool = True,
    bind_pose: BindPoseMode = "fit",
) -> SomaSkeletonIdentity:
    """Prepare only identity-dependent SOMA joint state."""
    xp = runtime.xp
    _, world_bind_pose = _prepare_bind_state(
        data,
        rest_shape_full=rest_shape_full,
        rest_shape_active=rest_shape_active,
        runtime=runtime,
        repose=repose,
        bind_pose=bind_pose,
    )
    return _prepare_skeleton_identity_state(xp, world_bind_pose, data.kinematics.kinematic_tree.parents)


def _prepare_bind_state(
    data: SomaWeights,
    *,
    rest_shape_full: Float[Array, "B Vf 3"],
    rest_shape_active: Float[Array, "B Va 3"],
    runtime: ArrayRuntime,
    repose: bool,
    bind_pose: BindPoseMode,
) -> tuple[Float[Array, "B Va 3"], Float[Array, "B Jf 4 4"]]:
    xp = runtime.xp
    control_rig = data.control_rig

    rest_shape_full, control_world_bind_pose_fit = _bind_pose_for_rest_shape(
        runtime=runtime,
        mode=bind_pose,
        bind_shape=data.bind_shape_full,
        bind_pose_world=control_rig.bind_pose_world,
        joint_regressor=control_rig.joint_regressor,
        joint_children_full=control_rig.joint_children_full,
        joint_children_indices_full=control_rig.joint_children_indices_full,
        skinned_vertex_indices_full=control_rig.skinned_vertex_indices_full,
        skinned_vertex_indices_full_index=control_rig.skinned_vertex_indices_full_index,
        tree=control_rig.kinematics.kinematic_tree,
        rest_shape=rest_shape_full,
    )
    bind_shape_active = rest_shape_active
    control_world_bind_pose = control_world_bind_pose_fit
    if repose:
        bind_shape_active, control_world_bind_pose = repose_to_bind_pose(
            runtime=runtime,
            rest_shape=rest_shape_active,
            skin_weights=control_rig.skin_weights_active,
            world_bind_pose_fit=control_world_bind_pose_fit,
            bind_pose_local=control_rig.bind_pose_local,
            tree=control_rig.kinematics.kinematic_tree,
        )
        control_world_bind_pose = _pin_root_transform(xp, control_world_bind_pose)
    world_bind_pose = _expand_control_bind_pose(xp, data, control_world_bind_pose)

    return bind_shape_active, world_bind_pose


def _prepare_identity_state(
    xp: Any,
    bind_shape: Float[Array, "*batch Va 3"],
    world_bind_pose: Float[Array, "*batch Jf 4 4"],
    parents_full: Sequence[int],
) -> SomaIdentity:
    identity = _prepare_skeleton_identity_state(xp, world_bind_pose, parents_full)
    inverse_bind_transforms = common.invert_rigid_transforms(world_bind_pose, xp=xp)
    inverse_bind_transforms = common.at_set(
        inverse_bind_transforms,
        (..., slice(None, 3), 3),
        inverse_bind_transforms[..., :3, 3] * 0.01,
        xp=xp,
    )
    return {
        "local_joint_translations": identity["local_joint_translations"],
        "rest_vertices": bind_shape * 0.01,
        "inverse_bind_transforms": inverse_bind_transforms,
    }


def _prepare_skeleton_identity_state(
    xp: Any,
    world_bind_pose: Float[Array, "*batch Jf 4 4"],
    parents_full: Sequence[int],
) -> SomaSkeletonIdentity:
    bind_local = _joint_world_to_local(xp, world_bind_pose, parents_full)
    local_joint_translations = bind_local[..., :3, 3]
    zeros = common.zeros_as(
        local_joint_translations,
        shape=(*local_joint_translations.shape[:-2], 3),
        xp=xp,
    )
    local_joint_translations = common.at_set(
        local_joint_translations,
        (..., 1, slice(None)),
        zeros,
        copy=False,
        xp=xp,
    )

    return {"local_joint_translations": local_joint_translations * 0.01}


def _pin_root_transform(
    xp: Any,
    transforms: Float[Array, "*batch J 4 4"],
) -> Float[Array, "*batch J 4 4"]:
    eye = common.eye_as(transforms, batch_dims=transforms.shape[:-3], xp=xp)
    return common.at_set(transforms, (..., 0, slice(None), slice(None)), eye, xp=xp)


def _bind_pose_for_rest_shape(
    *,
    runtime: ArrayRuntime,
    mode: BindPoseMode,
    bind_shape: Float[Array, "V 3"],
    bind_pose_world: Float[Array, "J 4 4"],
    joint_regressor: Float[Array, "J V"],
    joint_children_full: list[list[int]],
    joint_children_indices_full: Int[Array, "J C"],
    skinned_vertex_indices_full: list[list[int]],
    skinned_vertex_indices_full_index: Int[Array, "J K"],
    tree: common.KinematicTree,
    rest_shape: Float[Array, "B V 3"],
) -> tuple[Float[Array, "B V 3"], Float[Array, "B J 4 4"]]:
    xp = runtime.xp
    if mode not in ("fit", "fit_detached", "canonical"):
        raise ValueError(f"Unknown SOMA bind_pose mode: {mode!r}.")

    if mode == "canonical":
        batch_shape = rest_shape.shape[:-2]
        world_bind_pose = xp.broadcast_to(bind_pose_world, (*batch_shape, *bind_pose_world.shape))
        return rest_shape, world_bind_pose

    rest_shape, world_bind_pose = _fit_rest_shape_to_bind_pose(
        xp=xp,
        bind_shape=bind_shape,
        bind_pose_world=bind_pose_world,
        joint_regressor=joint_regressor,
        joint_children_full=joint_children_full,
        joint_children_indices_full=joint_children_indices_full,
        skinned_vertex_indices_full=skinned_vertex_indices_full,
        skinned_vertex_indices_full_index=skinned_vertex_indices_full_index,
        parents_full=tree.parents,
        rest_shape=rest_shape,
    )
    if mode == "fit_detached":
        rest_shape = runtime.stop_gradient(rest_shape)
        world_bind_pose = runtime.stop_gradient(world_bind_pose)
    return rest_shape, world_bind_pose


def _expand_control_bind_pose(
    xp: Any,
    data: SomaWeights,
    control_world_bind_pose: Float[Array, "*batch Jp 4 4"],
) -> Float[Array, "*batch Jf 4 4"]:
    control_indices = xp.asarray(data.control_rig.procedural.control_joint_indices_full)
    batch_shape = control_world_bind_pose.shape[:-3]
    internal_bind_pose = xp.asarray(data.bind_pose_world, dtype=control_world_bind_pose.dtype)
    target = xp.broadcast_to(internal_bind_pose, (*batch_shape, *internal_bind_pose.shape))
    target = common.at_set(target, (..., control_indices, slice(None), slice(None)), control_world_bind_pose, xp=xp)
    translations = xp.asarray(data.control_rig.procedural.translation_matrix, dtype=target.dtype) @ target[..., :3, 3]
    return common.at_set(target, (..., slice(None), slice(None, 3), 3), translations, xp=xp)


def prepare_pose(
    runtime: ArrayRuntime,
    data: SomaWeights,
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    rotation_type: RotationType,
    *,
    local_joint_translations: Float[Array, "*batch Jf 3"],
    inverse_bind_transforms: Float[Array, "*batch Jf 4 4"],
) -> common.deformation.SkinningPose:
    """Precompute pose-dependent SOMA state for repeated forward passes."""
    xp = runtime.xp
    correctives_pose_rot, skeleton_transforms_full = _prepare_skeleton_state(
        runtime,
        data,
        pose,
        rotation_type,
        local_joint_translations=local_joint_translations,
    )
    skinning_transforms = skeleton_transforms_full @ inverse_bind_transforms
    hidden = _corrective_hidden_activations(
        correctives_pose_rot,
        data.correctives.corrective_bindpose,
        data.correctives.hidden_weights,
        xp=xp,
    )
    return {
        "skeleton_transforms": _control_joint_transforms(xp, data, skeleton_transforms_full),
        "skinning_transforms": skinning_transforms[..., 1:, :, :],
        "pose_coefficients": hidden,
    }


def _corrective_hidden_activations(
    pose_rotations: Float[Array, "*batch J 3 3"],
    bindpose: Float[Array, "J 3 3"],
    weights: Float[Array, "input hidden"],
    *,
    xp: Any,
) -> Float[Array, "*batch H"]:
    """Evaluate SOMA's pose features and rectified hidden layer."""
    batch_shape = pose_rotations.shape[:-3]
    relative = bindpose.mT @ pose_rotations
    features = relative[..., :, :, :2]
    features = common.at_set(features, (..., slice(None), 0, 0), features[..., :, 0, 0] - 1, xp=xp)
    features = common.at_set(features, (..., slice(None), 1, 1), features[..., :, 1, 1] - 1, xp=xp)
    features = features.reshape(*batch_shape, -1)
    hidden = features @ weights
    return xp.maximum(hidden, xp.zeros_like(hidden))


def prepare_skeleton(
    runtime: ArrayRuntime,
    data: SomaWeights,
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    rotation_type: RotationType,
    *,
    local_joint_translations: Float[Array, "*batch Jf 3"],
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed SOMA public-joint transforms."""
    _, skeleton = _prepare_skeleton_state(
        runtime,
        data,
        pose,
        rotation_type,
        local_joint_translations=local_joint_translations,
    )
    return _control_joint_transforms(runtime.xp, data, skeleton)


def _prepare_skeleton_state(
    runtime: ArrayRuntime,
    data: SomaWeights,
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    rotation_type: RotationType,
    *,
    local_joint_translations: Float[Array, "*batch Jf 3"],
) -> tuple[
    Float[Array, "*batch Jp 3 3"],
    Float[Array, "*batch Jf 4 4"],
]:
    xp = runtime.xp
    pose_rot_control = SO3.convert(pose, src=rotation_type, dst="rotmat", xp=xp)
    pose_rot_full, control_local_rotations = _expand_control_pose_rotations(runtime, data, pose_rot_control)
    skeleton = _pose_skeleton(
        runtime,
        local_joint_translations,
        data.kinematics.kinematic_tree,
        pose_rot_full,
    )
    return control_local_rotations, skeleton


def _control_joint_transforms(
    xp, data: SomaWeights, transforms_full: Float[Array, "*batch Jf 4 4"]
) -> Float[Array, "*batch J 4 4"]:
    control_joint_indices = data.control_rig.procedural.control_joint_indices_full
    indices = xp.asarray(control_joint_indices[1:])
    return transforms_full[..., indices, :, :]


def _expand_control_pose_rotations(
    runtime: ArrayRuntime, data: SomaWeights, pose_rot: Float[Array, "*batch J 3 3"]
) -> tuple[Float[Array, "*batch Jf 3 3"], Float[Array, "*batch Jp 3 3"]]:
    xp = runtime.xp
    control_rig = data.control_rig
    procedural = control_rig.procedural
    control_joint_indices = procedural.control_joint_indices_full
    batch_shape = pose_rot.shape[:-3]
    root_identity = common.eye_as(pose_rot, batch_dims=(*batch_shape, 1), xp=xp)
    pose_rot_control = xp.concat([root_identity, pose_rot], axis=-3)
    control_local_rotations = _orient_pose_rot_full(
        xp,
        pose_rot,
        control_rig.t_pose_world,
        control_rig.kinematics.orientation_parent_indices,
    )
    control_local_translations = xp.asarray(control_rig.t_pose_local[..., :3, 3], dtype=pose_rot.dtype)
    control_world_transforms = _pose_skeleton(
        runtime,
        control_local_translations,
        control_rig.kinematics.kinematic_tree,
        control_local_rotations,
    )

    internal_joint_count = len(data.kinematics.kinematic_tree.parents)
    pose_rot_internal = common.eye_as(pose_rot, batch_dims=(*batch_shape, internal_joint_count), xp=xp)
    pose_rot_internal = common.at_set(
        pose_rot_internal,
        (..., xp.asarray(control_joint_indices), slice(None), slice(None)),
        pose_rot_control,
        xp=xp,
    )
    pose_rot_internal = _orient_pose_rot_full(
        xp,
        pose_rot_internal[..., 1:, :, :],
        data.t_pose_world,
        data.kinematics.orientation_parent_indices,
    )

    twist_values = _aligned_twist_channels_from_world(
        xp,
        data,
        control_world_transforms[..., :3, :3],
    )
    rotation_matrix = xp.asarray(procedural.rotation_matrix, dtype=pose_rot.dtype)
    twist_angles = twist_values @ rotation_matrix.mT
    twist_rot = _single_axis_rotation_matrices(
        xp,
        twist_angles,
        xp.asarray(procedural.twist_axis_ids),
        xp.asarray(procedural.twist_axis_signs, dtype=pose_rot.dtype),
    )
    twist_indices = xp.asarray(procedural.twist_joint_indices)
    current_twist_rot = pose_rot_internal[..., twist_indices, :, :]
    pose_rot_internal = common.at_set(
        pose_rot_internal,
        (..., twist_indices, slice(None), slice(None)),
        current_twist_rot @ twist_rot,
        xp=xp,
    )
    return pose_rot_internal, control_local_rotations


def _x_swing_twist_angles(xp, rotations: Float[Array, "... 3 3"]) -> Float[Array, "..."]:
    m00 = rotations[..., 0, 0]
    m11 = rotations[..., 1, 1]
    m12 = rotations[..., 1, 2]
    m21 = rotations[..., 2, 1]
    m22 = rotations[..., 2, 2]
    zero = xp.zeros_like(m00)
    eps = xp.full_like(m00, 1e-12)

    qw = 0.5 * xp.sqrt(xp.maximum(1.0 + m00 + m11 + m22, zero) + eps)
    qx = 0.5 * xp.copysign(
        xp.sqrt(xp.maximum(1.0 + m00 - m11 - m22, zero) + eps),
        m21 - m12,
    )
    return 4.0 * xp.atan2(qx, qw + 1.0)


def _aligned_twist_channels_from_world(
    xp,
    data: SomaWeights,
    control_world_rotations: Float[Array, "*batch Jp 3 3"],
) -> Float[Array, "*batch Jp"]:
    control_rig = data.control_rig
    procedural = control_rig.procedural
    start_ids = xp.asarray(procedural.segment_start_joint_indices)
    end_ids = xp.asarray(procedural.segment_end_joint_indices)
    parent_ids = xp.asarray(procedural.segment_parent_joint_indices)
    alignment = xp.asarray(
        procedural.segment_alignment_rotations,
        dtype=control_world_rotations.dtype,
    )
    bind_rotations = xp.asarray(
        control_rig.t_pose_world[..., :3, :3],
        dtype=control_world_rotations.dtype,
    )

    def virtual_rotations(joint_ids):
        current = control_world_rotations[..., joint_ids, :, :]
        return current @ bind_rotations[joint_ids].mT @ alignment

    end_virtual = virtual_rotations(end_ids)
    start_virtual = virtual_rotations(start_ids)
    parent_virtual = virtual_rotations(parent_ids)
    local_twist = _x_swing_twist_angles(xp, start_virtual.mT @ end_virtual)
    inherited_twist = _x_swing_twist_angles(xp, parent_virtual.mT @ start_virtual)

    twist_values = common.zeros_as(
        control_world_rotations,
        shape=(*control_world_rotations.shape[:-3], control_world_rotations.shape[-3]),
        xp=xp,
    )
    twist_values = common.at_set(twist_values, (..., end_ids), local_twist, xp=xp)
    reverse_indices = xp.asarray(procedural.segment_reverse_indices)
    reverse_start_ids = start_ids[reverse_indices]
    reverse_twist = inherited_twist[..., reverse_indices]
    return common.at_set(twist_values, (..., reverse_start_ids), reverse_twist, xp=xp)


def _single_axis_rotation_matrices(
    xp,
    angles: Float[Array, "*batch T"],
    axis_ids: Int[Array, "T"],
    axis_signs: Float[Array, "T"],
) -> Float[Array, "*batch T 3 3"]:
    angles = angles * axis_signs
    c = xp.cos(angles)
    s = xp.sin(angles)
    zeros = xp.zeros_like(angles)
    ones = xp.ones_like(angles)
    rx = xp.stack(
        [
            xp.stack([ones, zeros, zeros], axis=-1),
            xp.stack([zeros, c, -s], axis=-1),
            xp.stack([zeros, s, c], axis=-1),
        ],
        axis=-2,
    )
    ry = xp.stack(
        [
            xp.stack([c, zeros, s], axis=-1),
            xp.stack([zeros, ones, zeros], axis=-1),
            xp.stack([-s, zeros, c], axis=-1),
        ],
        axis=-2,
    )
    rz = xp.stack(
        [
            xp.stack([c, -s, zeros], axis=-1),
            xp.stack([s, c, zeros], axis=-1),
            xp.stack([zeros, zeros, ones], axis=-1),
        ],
        axis=-2,
    )
    matrices = xp.stack([rx, ry, rz], axis=-3)
    gather = axis_ids.reshape(*((1,) * (matrices.ndim - 4)), -1, 1, 1, 1)
    gather = xp.broadcast_to(gather, (*matrices.shape[:-4], matrices.shape[-4], 1, 3, 3))
    return common.take_along_axis(matrices, gather, axis=-3, xp=xp)[..., 0, :, :]


def fit_rigid_transform(
    source_points: Float[Array, "V 3"],
    target_points: Float[Array, "V 3"],
    *,
    xp: Any,
) -> tuple[Float[Array, "3 3"], Float[Array, "3"]]:
    source_center = xp.mean(source_points, axis=0)
    target_center = xp.mean(target_points, axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    covariance = source_centered.mT @ target_centered
    U, _S, Vh = xp.linalg.svd(covariance)
    reflection = common.eye_as(covariance, batch_dims=(), xp=xp)
    det = xp.linalg.det(Vh.mT @ U.mT)
    reflection = common.at_set(reflection, (-1, -1), xp.where(det < 0, -1.0, 1.0), xp=xp)
    rotation = Vh.mT @ reflection @ U.mT
    translation = target_center - source_center @ rotation.mT
    return rotation, translation


def repose_to_bind_pose(
    runtime: ArrayRuntime,
    rest_shape: Float[Array, "B V 3"],
    skin_weights: Float[Array, "V J"],
    world_bind_pose_fit: Float[Array, "B J 4 4"],
    bind_pose_local: Float[Array, "J 4 4"],
    tree: common.KinematicTree,
) -> tuple[Float[Array, "B V 3"], Float[Array, "B J 4 4"]]:
    xp = runtime.xp
    T_world = _repose_skeleton_to_bind_pose(
        runtime=runtime,
        world_bind_pose_fit=world_bind_pose_fit,
        bind_pose_local=bind_pose_local,
        tree=tree,
    )
    bone = T_world @ common.invert_rigid_transforms(world_bind_pose_fit, xp=xp)
    verts = skinning.linear_blend_skinning(rest_shape, bone, skin_weights, xp=xp)
    return verts, T_world


def identity_to_rest_vertices(
    xp,
    mean: Float[Array, "V 3"],
    shapedirs: Float[Array, "S V 3"],
    eigenvalues: Float[Array, "S"],
    identity: Float[Array, "B S"],
) -> Float[Array, "B V 3"]:
    coeffs = identity * xp.sqrt(eigenvalues)
    return mean + xp.einsum("...s,svc->...vc", coeffs, shapedirs)


def apply_rigid_transform(
    points: Float[Array, "... 3"],
    *,
    rotation: Float[Array, "3 3"],
    translation: Float[Array, "3"] | None = None,
    xp: Any,
) -> Float[Array, "... 3"]:
    transformed = points @ rotation.mT
    if translation is not None:
        transformed = transformed + translation
    return transformed


def _fit_rest_shape_to_bind_pose(
    xp,
    bind_shape: Float[Array, "V 3"],
    bind_pose_world: Float[Array, "J 4 4"],
    joint_regressor: Float[Array, "J V"],
    joint_children_full: list[list[int]],
    joint_children_indices_full: Int[Array, "J C"],
    skinned_vertex_indices_full: list[list[int]],
    skinned_vertex_indices_full_index: Int[Array, "J K"],
    parents_full: Sequence[int],
    rest_shape: Float[Array, "B V 3"],
) -> tuple[Float[Array, "B V 3"], Float[Array, "B J 4 4"]]:
    joint_positions = xp.einsum("jv,...vc->...jc", joint_regressor, rest_shape)
    world_bind_pose = _fit_joint_rotations(
        xp=xp,
        bind_shape=bind_shape,
        bind_pose_world=bind_pose_world,
        joint_children_full=joint_children_full,
        joint_children_indices_full=joint_children_indices_full,
        skinned_vertex_indices_full=skinned_vertex_indices_full,
        skinned_vertex_indices_full_index=skinned_vertex_indices_full_index,
        parents_full=parents_full,
        joint_positions=joint_positions,
        target_shape=rest_shape,
    )
    return rest_shape, world_bind_pose


def _repose_skeleton_to_bind_pose(
    runtime: ArrayRuntime,
    world_bind_pose_fit: Float[Array, "B J 4 4"],
    bind_pose_local: Float[Array, "J 4 4"],
    tree: common.KinematicTree,
) -> Float[Array, "B J 4 4"]:
    xp = runtime.xp
    batch_shape = world_bind_pose_fit.shape[:-3]
    bind_local_fit = _joint_world_to_local(xp, world_bind_pose_fit, tree.parents)
    local_t = bind_local_fit[..., :3, 3]

    zeros = xp.asarray(0.0, dtype=local_t.dtype)
    local_t = common.at_set(local_t, (..., 1, 0), zeros, copy=False, xp=xp)
    local_t = common.at_set(local_t, (..., 1, 2), zeros, copy=False, xp=xp)

    bind_rot = xp.broadcast_to(bind_pose_local[:, :3, :3], (*batch_shape, bind_pose_local.shape[0], 3, 3))
    T_local = common.affine_transforms(bind_rot, local_t, xp=xp)
    T_world = runtime._compose_kinematic_tree(T_local, tree)

    y_shift = xp.amin(T_world[..., :, 1, 3], axis=-1)
    return common.at_set(
        T_world,
        (..., slice(None), 1, 3),
        T_world[..., :, 1, 3] - y_shift[..., None],
        xp=xp,
    )


def _orient_pose_rot_full(
    xp,
    pose_rot: Float[Array, "B J 3 3"],
    t_pose_world: Float[Array, "Jf 4 4"],
    orientation_parent_indices: Int[Array, "Jf"],
) -> Float[Array, "B Jf 3 3"]:
    batch_shape = pose_rot.shape[:-3]
    root_identity = common.eye_as(pose_rot, batch_dims=(*batch_shape, 1), xp=xp)
    pose_rot_full = xp.concat([root_identity, pose_rot], axis=-3)
    orient = t_pose_world[:, :3, :3]
    orient_parent_T = orient[orientation_parent_indices].mT
    return orient_parent_T @ pose_rot_full @ orient


def _pose_skeleton(
    runtime: ArrayRuntime,
    local_joint_translations: Float[Array, "B Jf 3"],
    tree: common.KinematicTree,
    pose_rot_full: Float[Array, "B Jf 3 3"],
) -> Float[Array, "B Jf 4 4"]:
    xp = runtime.xp
    T_local = common.affine_transforms(pose_rot_full, local_joint_translations, xp=xp)
    return runtime._compose_kinematic_tree(T_local, tree)


def _fit_joint_rotations(
    xp,
    bind_shape: Float[Array, "V 3"],
    bind_pose_world: Float[Array, "J 4 4"],
    joint_children_full: list[list[int]],
    joint_children_indices_full: Int[Array, "J C"],
    skinned_vertex_indices_full: list[list[int]],
    skinned_vertex_indices_full_index: Int[Array, "J K"],
    parents_full: Sequence[int],
    joint_positions: Float[Array, "B J 3"],
    target_shape: Float[Array, "B V 3"],
) -> Float[Array, "B J 4 4"]:
    batch_shape = joint_positions.shape[:-2]
    J = joint_positions.shape[-2]
    bind_rot = bind_pose_world[:, :3, :3]
    bind_pos = bind_pose_world[:, :3, 3]

    rotations = [xp.broadcast_to(bind_rot[0], (*batch_shape, 3, 3))]
    for joint_index in range(1, J):
        children = joint_children_full[joint_index]
        if not children:
            parent_index = parents_full[joint_index]
            rotations.append(rotations[parent_index])
            continue

        skinned_vids = skinned_vertex_indices_full[joint_index]
        if skinned_vids:
            skinned_idx = skinned_vertex_indices_full_index[joint_index, : len(skinned_vids)]
            skinned_orig = bind_shape[skinned_idx] - bind_pos[joint_index]
            skinned_new = target_shape[..., skinned_idx, :] - joint_positions[..., joint_index : joint_index + 1, :]
            R_init = _align_vectors(
                xp,
                skinned_new,
                skinned_orig,
            )
        else:
            R_init = common.eye_as(bind_rot, batch_dims=batch_shape, xp=xp)

        child_idx = joint_children_indices_full[joint_index, : len(children)]
        pos_children_orig = bind_pos[child_idx] - bind_pos[joint_index : joint_index + 1]
        pos_children_orig = xp.einsum("...ij,cj->...ci", R_init, pos_children_orig)
        pos_children_new = joint_positions[..., child_idx, :] - joint_positions[..., joint_index : joint_index + 1, :]
        align_rot = _align_vectors(
            xp,
            pos_children_new,
            pos_children_orig,
        )
        R_joint = align_rot @ R_init @ bind_rot[joint_index]
        rotations.append(R_joint)

    R = xp.stack(rotations, axis=-3)
    return common.affine_transforms(R, joint_positions, xp=xp)


def _align_vectors(
    xp,
    target: Float[Array, "B N 3"],
    source: Float[Array, "B N 3"],
) -> Float[Array, "B 3 3"]:
    if target.shape[-2] == 1:
        return common.rotation_between_vectors(source[..., 0, :], target[..., 0, :], xp=xp)

    covariance = xp.einsum("...ni,...nj->...ij", target, source)
    return _kabsch(xp, covariance)


def _kabsch(xp, H: Float[Array, "B 3 3"]) -> Float[Array, "B 3 3"]:
    U, _, Vh = xp.linalg.svd(H)
    UVt = U @ Vh.mT
    det_sign = xp.where(_det3(UVt) < 0, xp.asarray(-1.0, dtype=H.dtype), xp.asarray(1.0, dtype=H.dtype))
    D = common.eye_as(H, batch_dims=H.shape[:-2], xp=xp)
    D = common.at_set(D, (..., 2, 2), det_sign, xp=xp)
    return U @ D @ Vh


def _det3(M: Float[Array, "B 3 3"]) -> Float[Array, "B"]:
    return (
        M[..., 0, 0] * (M[..., 1, 1] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 1])
        - M[..., 0, 1] * (M[..., 1, 0] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 0])
        + M[..., 0, 2] * (M[..., 1, 0] * M[..., 2, 1] - M[..., 1, 1] * M[..., 2, 0])
    )


def _joint_world_to_local(
    xp,
    world: Float[Array, "B J 4 4"],
    parents_full: Sequence[int],
) -> Float[Array, "B J 4 4"]:
    inv = common.invert_rigid_transforms(world, xp=xp)
    local = inv[..., xp.asarray(parents_full), :, :] @ world
    for joint, parent in enumerate(parents_full):
        if joint == parent:
            local = common.at_set(local, (..., joint, slice(None), slice(None)), world[..., joint, :, :], xp=xp)
    return local
