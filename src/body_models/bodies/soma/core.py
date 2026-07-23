"""SOMA identity, rigging, and pose mathematics."""

from __future__ import annotations
import contextlib
from typing import Any, Literal, TypedDict

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import common
from body_models.bodies.soma.correctives import CorrectiveNetwork, hidden_activations
from body_models.common import skinning
from body_models.rotations import RotationType

Array = Any
Front = tuple[list[int], list[int]]
BindPoseMode = Literal["fit", "fit_detached", "canonical"]


class SomaSkeletonIdentity(TypedDict):
    """Identity-dependent joint state needed to pose the SOMA skeleton."""

    local_joint_translations: Float[Array, "*batch Jf 3"]


class SomaIdentity(SomaSkeletonIdentity):
    """Complete identity-dependent SOMA mesh state."""

    rest_vertices: Float[Array, "*batch Va 3"]
    inverse_bind_transforms: Float[Array, "*batch Jf 4 4"]


class SomaPreparedPose(TypedDict):
    """Complete pose-dependent SOMA mesh state."""

    skeleton_transforms: Float[Array, "*batch J 4 4"]
    skinning_transforms: Float[Array, "*batch J 4 4"]
    pose_offsets: Float[Array, "*batch Va 3"]


def skinning_weights(data: Any) -> Float[Array, "Va Jf"]:
    return data.skin_weights_active[:, 1:]


def prepare_identity_from_rest_shape(
    data: Any,
    *,
    rest_shape_full: Float[Array, "B Vf 3"],
    rest_shape_active: Float[Array, "B Va 3"],
    match_warp: bool,
    xp: Any,
    repose: bool = True,
    bind_pose: BindPoseMode = "fit",
) -> SomaIdentity:
    bind_shape, world_bind_pose = _prepare_bind_state(
        data,
        rest_shape_full=rest_shape_full,
        rest_shape_active=rest_shape_active,
        match_warp=match_warp,
        xp=xp,
        repose=repose,
        bind_pose=bind_pose,
    )
    return _prepare_identity_state(xp, bind_shape, world_bind_pose, data.topology.parents_full)


def prepare_skeleton_identity_from_rest_shape(
    data: Any,
    *,
    rest_shape_full: Float[Array, "B Vf 3"],
    rest_shape_active: Float[Array, "B Va 3"],
    match_warp: bool,
    xp: Any,
    repose: bool = True,
    bind_pose: BindPoseMode = "fit",
) -> SomaSkeletonIdentity:
    """Prepare only identity-dependent SOMA joint state."""
    _, world_bind_pose = _prepare_bind_state(
        data,
        rest_shape_full=rest_shape_full,
        rest_shape_active=rest_shape_active,
        match_warp=match_warp,
        xp=xp,
        repose=repose,
        bind_pose=bind_pose,
    )
    return _prepare_skeleton_identity_state(xp, world_bind_pose, data.topology.parents_full)


def _prepare_bind_state(
    data: Any,
    *,
    rest_shape_full: Float[Array, "B Vf 3"],
    rest_shape_active: Float[Array, "B Va 3"],
    match_warp: bool,
    xp: Any,
    repose: bool,
    bind_pose: BindPoseMode,
) -> tuple[Float[Array, "B Va 3"], Float[Array, "B Jf 4 4"]]:
    if data.public is not None:
        return _prepare_procedural_bind_state(
            data,
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
            match_warp=match_warp,
            xp=xp,
            repose=repose,
            bind_pose=bind_pose,
        )

    rest_shape_full, world_bind_pose_fit = _bind_pose_for_rest_shape(
        xp=xp,
        mode=bind_pose,
        bind_shape=data.bind_shape_full,
        bind_pose_world=data.bind_pose_world,
        joint_regressor=data.joint_regressor,
        joint_children_full=data.topology.joint_children_full,
        joint_children_indices_full=data.topology.joint_children_indices_full,
        skinned_vertex_indices_full=data.topology.skinned_vertex_indices_full,
        skinned_vertex_indices_full_index=data.topology.skinned_vertex_indices_full_index,
        parents_full=data.topology.parents_full,
        rest_shape=rest_shape_full,
        match_warp=match_warp,
    )
    bind_shape_active = rest_shape_active
    world_bind_pose = world_bind_pose_fit
    if repose:
        bind_shape_active, world_bind_pose = repose_to_bind_pose(
            xp=xp,
            rest_shape=rest_shape_active,
            skin_weights=data.skin_weights_active,
            world_bind_pose_fit=world_bind_pose_fit,
            bind_pose_local=data.bind_pose_local,
            kinematic_fronts=data.topology.kinematic_fronts_full,
            parents_full=data.topology.parents_full,
        )
    return bind_shape_active, world_bind_pose


def _prepare_procedural_bind_state(
    data: Any,
    *,
    rest_shape_full: Float[Array, "B Vf 3"],
    rest_shape_active: Float[Array, "B Va 3"],
    match_warp: bool,
    xp: Any,
    repose: bool,
    bind_pose: BindPoseMode,
) -> tuple[Float[Array, "B Va 3"], Float[Array, "B Jf 4 4"]]:
    public = data.public

    rest_shape_full, public_world_bind_pose_fit = _bind_pose_for_rest_shape(
        xp=xp,
        mode=bind_pose,
        bind_shape=data.bind_shape_full,
        bind_pose_world=public.bind_pose_world,
        joint_regressor=public.joint_regressor,
        joint_children_full=public.topology.joint_children_full,
        joint_children_indices_full=public.topology.joint_children_indices_full,
        skinned_vertex_indices_full=public.topology.skinned_vertex_indices_full,
        skinned_vertex_indices_full_index=public.topology.skinned_vertex_indices_full_index,
        parents_full=public.topology.parents_full,
        rest_shape=rest_shape_full,
        match_warp=match_warp,
    )
    bind_shape_active = rest_shape_active
    public_world_bind_pose = public_world_bind_pose_fit
    if repose:
        bind_shape_active, public_world_bind_pose = repose_to_bind_pose(
            xp=xp,
            rest_shape=rest_shape_active,
            skin_weights=public.skin_weights_active,
            world_bind_pose_fit=public_world_bind_pose_fit,
            bind_pose_local=public.bind_pose_local,
            kinematic_fronts=public.topology.kinematic_fronts_full,
            parents_full=public.topology.parents_full,
        )
        public_world_bind_pose = _pin_root_transform(xp, public_world_bind_pose)
    world_bind_pose = _expand_public_bind_pose(xp, data, public_world_bind_pose)

    return bind_shape_active, world_bind_pose


def _prepare_identity_state(
    xp: Any,
    bind_shape: Float[Array, "*batch Va 3"],
    world_bind_pose: Float[Array, "*batch Jf 4 4"],
    parents_full: list[int],
) -> SomaIdentity:
    identity = _prepare_skeleton_identity_state(xp, world_bind_pose, parents_full)
    inverse_bind_transforms = common.invert_rigid_transforms(world_bind_pose, xp=xp)
    inverse_bind_transforms = common.set(
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
    parents_full: list[int],
) -> SomaSkeletonIdentity:
    bind_local = _joint_world_to_local(xp, world_bind_pose, parents_full)
    local_joint_translations = bind_local[..., :3, 3]
    zeros = common.zeros_as(
        local_joint_translations,
        shape=(*local_joint_translations.shape[:-2], 3),
        xp=xp,
    )
    local_joint_translations = common.set(
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
    return common.set(transforms, (..., 0, slice(None), slice(None)), eye, xp=xp)


def _stop_gradient(xp: Any, x: Float[Array, "..."]) -> Float[Array, "..."]:
    name = xp.__name__
    if name == "jax.numpy":
        import jax

        return jax.lax.stop_gradient(x)
    if name == "numpy":
        return x
    raise NotImplementedError(f'bind_pose="fit_detached" is not implemented for {name}.')


def _bind_pose_for_rest_shape(
    *,
    xp: Any,
    mode: BindPoseMode,
    bind_shape: Float[Array, "V 3"],
    bind_pose_world: Float[Array, "J 4 4"],
    joint_regressor: Float[Array, "J V"],
    joint_children_full: list[list[int]],
    joint_children_indices_full: Int[Array, "J C"],
    skinned_vertex_indices_full: list[list[int]],
    skinned_vertex_indices_full_index: Int[Array, "J K"],
    parents_full: list[int],
    rest_shape: Float[Array, "B V 3"],
    match_warp: bool,
) -> tuple[Float[Array, "B V 3"], Float[Array, "B J 4 4"]]:
    if mode not in ("fit", "fit_detached", "canonical"):
        raise ValueError(f"Unknown SOMA bind_pose mode: {mode!r}.")

    if mode == "canonical":
        batch_shape = rest_shape.shape[:-2]
        world_bind_pose = xp.broadcast_to(bind_pose_world, (*batch_shape, *bind_pose_world.shape))
        return rest_shape, world_bind_pose

    fit_context = contextlib.nullcontext()
    if mode == "fit_detached" and xp.__name__ == "torch":
        import torch

        fit_context = torch.no_grad()

    with fit_context:
        rest_shape, world_bind_pose = _fit_rest_shape_to_bind_pose(
            xp=xp,
            bind_shape=bind_shape,
            bind_pose_world=bind_pose_world,
            joint_regressor=joint_regressor,
            joint_children_full=joint_children_full,
            joint_children_indices_full=joint_children_indices_full,
            skinned_vertex_indices_full=skinned_vertex_indices_full,
            skinned_vertex_indices_full_index=skinned_vertex_indices_full_index,
            parents_full=parents_full,
            rest_shape=rest_shape,
            match_warp=match_warp,
        )
    if mode == "fit_detached" and xp.__name__ != "torch":
        world_bind_pose = _stop_gradient(xp, world_bind_pose)
    return rest_shape, world_bind_pose


def _expand_public_bind_pose(
    xp: Any,
    data: Any,
    public_world_bind_pose: Float[Array, "*batch Jp 4 4"],
) -> Float[Array, "*batch Jf 4 4"]:
    public_indices = xp.asarray(data.public.procedural.public_joint_indices_full)
    batch_shape = public_world_bind_pose.shape[:-3]
    internal_bind_pose = xp.asarray(data.bind_pose_world, dtype=public_world_bind_pose.dtype)
    target = xp.broadcast_to(internal_bind_pose, (*batch_shape, *internal_bind_pose.shape))
    target = common.set(target, (..., public_indices, slice(None), slice(None)), public_world_bind_pose, xp=xp)
    translations = xp.asarray(data.public.procedural.translation_matrix, dtype=target.dtype) @ target[..., :3, 3]
    return common.set(target, (..., slice(None), slice(None, 3), 3), translations, xp=xp)


def prepare_pose(
    data: Any,
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    rotation_type: RotationType,
    *,
    local_joint_translations: Float[Array, "*batch Jf 3"],
    inverse_bind_transforms: Float[Array, "*batch Jf 4 4"],
    xp: Any,
    corrective_network: CorrectiveNetwork,
) -> SomaPreparedPose:
    """Precompute pose-dependent SOMA state for repeated forward passes."""
    pose_rot_public, pose_rot_full, skeleton_transforms_full = _prepare_skeleton_state(
        data,
        pose,
        rotation_type,
        local_joint_translations=local_joint_translations,
        xp=xp,
    )
    skinning_transforms = skeleton_transforms_full @ inverse_bind_transforms
    correctives_pose_rot = pose_rot_full
    if data.public is not None:
        correctives_pose_rot = _orient_pose_rot_full(
            xp,
            pose_rot_public,
            data.public.t_pose_world,
            data.public.topology.parent_indices_full,
        )
    hidden = hidden_activations(
        correctives_pose_rot,
        data.correctives.corrective_bindpose,
        data.correctives.corrective_W1,
        xp=xp,
    )
    pose_offsets = corrective_network(hidden)
    if data.vertex_map is not None:
        pose_offsets = pose_offsets[..., data.vertex_map, :]
    return {
        "skeleton_transforms": _public_joint_transforms(xp, data, skeleton_transforms_full),
        "skinning_transforms": skinning_transforms[..., 1:, :, :],
        "pose_offsets": pose_offsets * 0.01,
    }


def prepare_skeleton(
    data: Any,
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    rotation_type: RotationType,
    *,
    local_joint_translations: Float[Array, "*batch Jf 3"],
    xp: Any,
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed SOMA public-joint transforms."""
    _, _, skeleton = _prepare_skeleton_state(
        data,
        pose,
        rotation_type,
        local_joint_translations=local_joint_translations,
        xp=xp,
    )
    return _public_joint_transforms(xp, data, skeleton)


def _prepare_skeleton_state(
    data: Any,
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    rotation_type: RotationType,
    *,
    local_joint_translations: Float[Array, "*batch Jf 3"],
    xp: Any,
) -> tuple[
    Float[Array, "*batch J 3 3"],
    Float[Array, "*batch Jf 3 3"],
    Float[Array, "*batch Jf 4 4"],
]:
    pose_rot_public = SO3.convert(pose, src=rotation_type, dst="rotmat", xp=xp)
    pose_rot_internal = _expand_public_pose_rotations(xp, data, pose_rot_public)
    pose_rot_full = _orient_pose_rot_full(
        xp,
        pose_rot_internal,
        data.t_pose_world,
        data.topology.parent_indices_full,
    )
    skeleton = _pose_skeleton(
        xp,
        local_joint_translations,
        data.topology.kinematic_fronts_full,
        pose_rot_full,
    )
    return pose_rot_public, pose_rot_full, skeleton


def _public_joint_transforms(
    xp, data: Any, transforms_full: Float[Array, "*batch Jf 4 4"]
) -> Float[Array, "*batch J 4 4"]:
    if data.public is None:
        return transforms_full[..., 1:, :, :]
    public_joint_indices = data.public.procedural.public_joint_indices_full
    indices = xp.asarray(public_joint_indices[1:])
    return transforms_full[..., indices, :, :]


def _expand_public_pose_rotations(
    xp, data: Any, pose_rot: Float[Array, "*batch J 3 3"]
) -> Float[Array, "*batch Ji 3 3"]:
    if data.public is None:
        return pose_rot

    procedural = data.public.procedural
    public_joint_indices = procedural.public_joint_indices_full
    batch_shape = pose_rot.shape[:-3]
    root_identity = common.eye_as(pose_rot, batch_dims=(*batch_shape, 1), xp=xp)
    pose_rot_public = xp.concat([root_identity, pose_rot], axis=-3)
    internal_joint_count = len(data.topology.parents_full)
    pose_rot_internal = common.eye_as(pose_rot, batch_dims=(*batch_shape, internal_joint_count), xp=xp)
    pose_rot_internal = common.set(
        pose_rot_internal,
        (..., xp.asarray(public_joint_indices), slice(None), slice(None)),
        pose_rot_public,
        xp=xp,
    )

    source_axis_ids = xp.asarray(procedural.source_axis_ids)
    source_axis_signs = xp.asarray(procedural.source_axis_signs, dtype=pose_rot.dtype)
    twist_values = _local_axis_twist_angles(xp, pose_rot_public, source_axis_ids) * source_axis_signs
    twist_angles = twist_values @ xp.asarray(procedural.rotation_matrix, dtype=pose_rot.dtype).swapaxes(-2, -1)
    twist_rot = _single_axis_rotation_matrices(
        xp,
        twist_angles,
        xp.asarray(procedural.twist_axis_ids),
        xp.asarray(procedural.twist_axis_signs, dtype=pose_rot.dtype),
    )
    twist_indices = xp.asarray(procedural.twist_joint_indices)
    current_twist_rot = pose_rot_internal[..., twist_indices, :, :]
    pose_rot_internal = common.set(
        pose_rot_internal,
        (..., twist_indices, slice(None), slice(None)),
        current_twist_rot @ twist_rot,
        xp=xp,
    )
    return pose_rot_internal[..., 1:, :, :]


def _local_axis_twist_angles(
    xp, rotations: Float[Array, "*batch J 3 3"], axis_ids: Int[Array, "J"]
) -> Float[Array, "*batch J"]:
    x = xp.atan2(rotations[..., :, 2, 1], rotations[..., :, 1, 1])
    y = xp.atan2(rotations[..., :, 0, 2], rotations[..., :, 0, 0])
    z = xp.atan2(rotations[..., :, 1, 0], rotations[..., :, 0, 0])
    angles = xp.stack([x, y, z], axis=-1)
    index = axis_ids.reshape(*((1,) * (angles.ndim - 2)), -1, 1)
    index = xp.broadcast_to(index, (*angles.shape[:-1], 1))
    return _take_along_axis(xp, angles, index, axis=-1)[..., 0]


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
    return _take_along_axis(xp, matrices, gather, axis=-3)[..., 0, :, :]


def _take_along_axis(
    xp: Any,
    array: Float[Array, "..."],
    indices: Int[Array, "..."],
    axis: int,
) -> Float[Array, "..."]:
    if hasattr(xp, "take_along_axis"):
        return xp.take_along_axis(array, indices, axis=axis)
    return xp.gather(array, dim=axis, index=indices)


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
    covariance = source_centered.swapaxes(-2, -1) @ target_centered
    U, _S, Vh = xp.linalg.svd(covariance)
    reflection = common.eye_as(covariance, batch_dims=(), xp=xp)
    det = xp.linalg.det(Vh.swapaxes(-2, -1) @ U.swapaxes(-2, -1))
    reflection = common.set(reflection, (-1, -1), xp.where(det < 0, -1.0, 1.0), xp=xp)
    rotation = Vh.swapaxes(-2, -1) @ reflection @ U.swapaxes(-2, -1)
    translation = target_center - source_center @ rotation.swapaxes(-2, -1)
    return rotation, translation


def repose_to_bind_pose(
    xp,
    rest_shape: Float[Array, "B V 3"],
    skin_weights: Float[Array, "V J"],
    world_bind_pose_fit: Float[Array, "B J 4 4"],
    bind_pose_local: Float[Array, "J 4 4"],
    kinematic_fronts: list[Front],
    parents_full: list[int],
) -> tuple[Float[Array, "B V 3"], Float[Array, "B J 4 4"]]:
    T_world = _repose_skeleton_to_bind_pose(
        xp=xp,
        world_bind_pose_fit=world_bind_pose_fit,
        bind_pose_local=bind_pose_local,
        kinematic_fronts=kinematic_fronts,
        parents_full=parents_full,
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
    transformed = points @ rotation.swapaxes(-2, -1)
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
    parents_full: list[int],
    rest_shape: Float[Array, "B V 3"],
    match_warp: bool,
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
        match_warp=match_warp,
    )
    return rest_shape, world_bind_pose


def _repose_skeleton_to_bind_pose(
    xp,
    world_bind_pose_fit: Float[Array, "B J 4 4"],
    bind_pose_local: Float[Array, "J 4 4"],
    kinematic_fronts: list[Front],
    parents_full: list[int],
) -> Float[Array, "B J 4 4"]:
    batch_shape = world_bind_pose_fit.shape[:-3]
    bind_local_fit = _joint_world_to_local(xp, world_bind_pose_fit, parents_full)
    local_t = bind_local_fit[..., :3, 3]

    zeros = xp.asarray(0.0, dtype=local_t.dtype)
    local_t = common.set(local_t, (..., 1, 0), zeros, copy=False, xp=xp)
    local_t = common.set(local_t, (..., 1, 2), zeros, copy=False, xp=xp)

    bind_rot = xp.broadcast_to(bind_pose_local[:, :3, :3], (*batch_shape, bind_pose_local.shape[0], 3, 3))
    T_local = common.affine_transforms(bind_rot, local_t, xp=xp)
    T_world = common.compose_local_transforms(T_local, kinematic_fronts, xp=xp)

    y_shift = xp.amin(T_world[..., :, 1, 3], axis=-1)
    return common.set(
        T_world,
        (..., slice(None), 1, 3),
        T_world[..., :, 1, 3] - y_shift[..., None],
        xp=xp,
    )


def _orient_pose_rot_full(
    xp,
    pose_rot: Float[Array, "B J 3 3"],
    t_pose_world: Float[Array, "Jf 4 4"],
    parent_indices_full: Int[Array, "Jf"],
) -> Float[Array, "B Jf 3 3"]:
    batch_shape = pose_rot.shape[:-3]
    root_identity = common.eye_as(pose_rot, batch_dims=(*batch_shape, 1), xp=xp)
    pose_rot_full = xp.concat([root_identity, pose_rot], axis=-3)
    orient = t_pose_world[:, :3, :3]
    orient_parent_T = orient[parent_indices_full].swapaxes(-2, -1)
    return orient_parent_T @ pose_rot_full @ orient


def _pose_skeleton(
    xp,
    local_joint_translations: Float[Array, "B Jf 3"],
    kinematic_fronts: list[Front],
    pose_rot_full: Float[Array, "B Jf 3 3"],
) -> Float[Array, "B Jf 4 4"]:
    T_local = common.affine_transforms(pose_rot_full, local_joint_translations, xp=xp)
    return common.compose_local_transforms(T_local, kinematic_fronts, xp=xp)


def _fit_joint_rotations(
    xp,
    bind_shape: Float[Array, "V 3"],
    bind_pose_world: Float[Array, "J 4 4"],
    joint_children_full: list[list[int]],
    joint_children_indices_full: Int[Array, "J C"],
    skinned_vertex_indices_full: list[list[int]],
    skinned_vertex_indices_full_index: Int[Array, "J K"],
    parents_full: list[int],
    joint_positions: Float[Array, "B J 3"],
    target_shape: Float[Array, "B V 3"],
    match_warp: bool,
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
                match_warp=match_warp,
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
            match_warp=match_warp,
        )
        R_joint = align_rot @ R_init @ bind_rot[joint_index]
        rotations.append(R_joint)

    R = xp.stack(rotations, axis=-3)
    return common.affine_transforms(R, joint_positions, xp=xp)


def _align_vectors(
    xp,
    target: Float[Array, "B N 3"],
    source: Float[Array, "B N 3"],
    *,
    match_warp: bool,
) -> Float[Array, "B 3 3"]:
    if target.shape[-2] == 1:
        return _rotation_between_vectors(xp, target[..., 0, :], source[..., 0, :])

    H = xp.einsum("...ni,...nj->...ij", target, source)
    # SOMALayer's default warp path uses plain covariance; the alternate path adds a virtual normal.
    if not match_warp:
        p0, p1 = target[..., 0, :], target[..., 1, :]
        q0, q1 = source[..., 0, :], source[..., 1, :]
        n_target = xp.linalg.cross(p0, p1)
        n_source = xp.linalg.cross(q0, q1)
        len_target = xp.linalg.vector_norm(n_target, axis=-1, keepdims=True)
        len_source = xp.linalg.vector_norm(n_source, axis=-1, keepdims=True)
        scale_target = xp.linalg.vector_norm(p0, axis=-1, keepdims=True) / (len_target + 1e-8)
        scale_source = xp.linalg.vector_norm(q0, axis=-1, keepdims=True) / (len_source + 1e-8)
        valid = (len_target[..., 0] > 1e-9) & (len_source[..., 0] > 1e-9)
        v_target = n_target * scale_target
        v_source = n_source * scale_source
        virtual = xp.einsum("...i,...j->...ij", v_target, v_source)
        H = xp.where(valid[..., None, None], H + virtual, H)
    return _kabsch(xp, H)


def _kabsch(xp, H: Float[Array, "B 3 3"]) -> Float[Array, "B 3 3"]:
    U, _, Vh = xp.linalg.svd(H)
    UVt = U @ Vh.swapaxes(-2, -1)
    det_sign = xp.where(_det3(UVt) < 0, xp.asarray(-1.0, dtype=H.dtype), xp.asarray(1.0, dtype=H.dtype))
    D = common.eye_as(H, batch_dims=H.shape[:-2], xp=xp)
    D = common.set(D, (..., 2, 2), det_sign, xp=xp)
    return U @ D @ Vh


def _rotation_between_vectors(
    xp,
    target: Float[Array, "B 3"],
    source: Float[Array, "B 3"],
) -> Float[Array, "B 3 3"]:
    target_norm = xp.linalg.vector_norm(target, axis=-1, keepdims=True)
    source_norm = xp.linalg.vector_norm(source, axis=-1, keepdims=True)
    target_unit = target / xp.where(target_norm > 1e-8, target_norm, xp.ones_like(target_norm))
    source_unit = source / xp.where(source_norm > 1e-8, source_norm, xp.ones_like(source_norm))

    dot = xp.sum(target_unit * source_unit, axis=-1, keepdims=True)
    dot = xp.clip(dot, -1.0, 1.0)
    cross = xp.linalg.cross(source_unit, target_unit)
    cross_norm = xp.linalg.vector_norm(cross, axis=-1, keepdims=True)
    axis = cross / xp.where(cross_norm > 1e-8, cross_norm, xp.ones_like(cross_norm))

    antiparallel = dot[..., 0] < -1.0 + 1e-6
    basis = common.eye_as(target, batch_dims=target.shape[:-1], xp=xp)
    x_vec = basis[..., 0, :]
    y_vec = basis[..., 1, :]
    w = xp.where(xp.abs(source_unit[..., 0:1]) > 0.6, y_vec, x_vec)
    antiparallel_axis = xp.linalg.cross(source_unit, w)
    antiparallel_axis_norm = xp.linalg.vector_norm(antiparallel_axis, axis=-1, keepdims=True)
    fallback_norm = xp.ones_like(antiparallel_axis_norm)
    antiparallel_axis_norm = xp.where(antiparallel_axis_norm > 1e-8, antiparallel_axis_norm, fallback_norm)
    antiparallel_axis = antiparallel_axis / antiparallel_axis_norm
    axis = xp.where(antiparallel[..., None], antiparallel_axis, axis)

    angle = xp.atan2(cross_norm[..., 0], dot[..., 0])
    pi = xp.asarray(3.141592653589793, dtype=target.dtype)
    angle = xp.where(antiparallel, pi, angle)
    return SO3.conversions.from_axis_angle_to_rotmat(angle[..., None] * axis, xp=xp)


def _det3(M: Float[Array, "B 3 3"]) -> Float[Array, "B"]:
    return (
        M[..., 0, 0] * (M[..., 1, 1] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 1])
        - M[..., 0, 1] * (M[..., 1, 0] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 0])
        + M[..., 0, 2] * (M[..., 1, 0] * M[..., 2, 1] - M[..., 1, 1] * M[..., 2, 0])
    )


def _joint_world_to_local(xp, world: Float[Array, "B J 4 4"], parents_full: list[int]) -> Float[Array, "B J 4 4"]:
    inv = common.invert_rigid_transforms(world, xp=xp)
    local = inv[..., xp.asarray(parents_full), :, :] @ world
    for joint, parent in enumerate(parents_full):
        if joint == parent:
            local = common.set(local, (..., joint, slice(None), slice(None)), world[..., joint, :, :], xp=xp)
    return local
