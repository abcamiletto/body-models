"""Backend-agnostic SOMA computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jaxtyping import Float, Int
from nanomanifold import SO3

from ... import common
from ...common import get_namespace
from ...rotations import RotationType

Array = Any
Front = tuple[list[int], list[int]]


@dataclass(frozen=True)
class PreparedSomaIdentity:
    rest_shape_full: Float[Array, "B Vf 3"]
    rest_shape_active: Float[Array, "B Va 3"]
    world_bind_pose_fit: Float[Array, "B Jf 4 4"]


def prepare_data(soma_weights: Any) -> Any:
    return soma_weights


def prepare_identity_from_rest_shape(
    data: Any,
    *,
    rest_shape_full: Float[Array, "B Vf 3"],
    rest_shape_active: Float[Array, "B Va 3"],
    match_warp: bool,
    xp: Any,
) -> PreparedSomaIdentity:
    rest_shape_full, world_bind_pose_fit = _fit_rest_shape_to_bind_pose(
        xp=xp,
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
    return PreparedSomaIdentity(rest_shape_full, rest_shape_active, world_bind_pose_fit)


def forward_vertices(
    data: Any,
    prepared_identity: PreparedSomaIdentity,
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    apply_correctives: bool = True,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any,
) -> Float[Array, "B V 3"]:
    return _forward_vertices_with(
        data=data,
        prepared_identity=prepared_identity,
        pose=pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        apply_correctives=apply_correctives,
        rotation_type=rotation_type,
        xp=xp,
        apply_pose_correctives_fn=apply_pose_correctives,
        linear_blend_skinning_fn=linear_blend_skinning,
    )


def _forward_vertices_with(
    data: Any,
    prepared_identity: PreparedSomaIdentity,
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    apply_correctives: bool = True,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any,
    apply_pose_correctives_fn: Any,
    linear_blend_skinning_fn: Any,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3] in meters."""
    pose_rot = SO3.convert(pose, src=rotation_type, dst="rotmat", xp=xp)
    topology = data.topology
    pose_rot_full = _orient_pose_rot_full(xp, pose_rot, data.t_pose_world, topology.parents_full)
    rest_shape_active = prepared_identity.rest_shape_active
    world_bind_pose_fit = prepared_identity.world_bind_pose_fit

    if apply_correctives:
        rest_shape_active, world_bind_pose = repose_to_bind_pose(
            xp=xp,
            rest_shape=rest_shape_active,
            skin_weights=data.skin_weights_active,
            world_bind_pose_fit=world_bind_pose_fit,
            bind_pose_local=data.bind_pose_local,
            kinematic_fronts=topology.kinematic_fronts_full,
            parents_full=topology.parents_full,
            linear_blend_skinning_fn=linear_blend_skinning_fn,
        )
        corrective_offsets = apply_pose_correctives_fn(data, pose_rot_full, xp=xp)
        if data.vertex_map is not None:
            corrective_offsets = corrective_offsets[:, data.vertex_map]
        rest_shape_active = rest_shape_active + corrective_offsets
    else:
        world_bind_pose = world_bind_pose_fit

    verts_cm, _ = pose_mesh_from_oriented_pose(
        xp=xp,
        rest_shape=rest_shape_active,
        skin_weights=data.skin_weights_active,
        world_bind_pose=world_bind_pose,
        kinematic_fronts=topology.kinematic_fronts_full,
        parents_full=topology.parents_full,
        pose_rot_full=pose_rot_full,
        linear_blend_skinning_fn=linear_blend_skinning_fn,
    )

    verts = verts_cm * 0.01
    verts = _apply_global_transform_vertices(xp, verts, global_rotation, global_translation, rotation_type)
    if vertex_indices is not None:
        verts = verts[:, xp.asarray(vertex_indices)]
    return verts


def forward_skeleton(
    data: Any,
    prepared_identity: PreparedSomaIdentity,
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    apply_correctives: bool = True,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any,
) -> Float[Array, "B J 4 4"]:
    """Compute skeleton transforms [B, J, 4, 4] in meters."""
    pose_rot = SO3.convert(pose, src=rotation_type, dst="rotmat", xp=xp)
    topology = data.topology
    world_bind_pose = prepared_identity.world_bind_pose_fit
    if apply_correctives:
        world_bind_pose = _repose_skeleton_to_bind_pose(
            xp=xp,
            world_bind_pose_fit=world_bind_pose,
            bind_pose_local=data.bind_pose_local,
            kinematic_fronts=topology.kinematic_fronts_full,
            parents_full=topology.parents_full,
        )
    pose_rot_full = _orient_pose_rot_full(xp, pose_rot, data.t_pose_world, topology.parents_full)
    T_world_cm = _pose_skeleton_from_oriented_pose(
        xp=xp,
        world_bind_pose=world_bind_pose,
        kinematic_fronts=topology.kinematic_fronts_full,
        parents_full=topology.parents_full,
        pose_rot_full=pose_rot_full,
    )

    T_world = common.set(T_world_cm, (..., slice(None, 3), 3), T_world_cm[..., :3, 3] * 0.01, xp=xp)
    T_world = _apply_global_transform_transforms(xp, T_world, global_rotation, global_translation, rotation_type)
    public = T_world[:, 1:]
    if joint_indices is not None:
        public = public[:, xp.asarray(joint_indices)]
    return public


def fit_rigid_transform(
    source_points: Float[Array, "V 3"],
    target_points: Float[Array, "V 3"],
    *,
    xp: Any = None,
) -> tuple[Float[Array, "3 3"], Float[Array, "3"]]:
    if xp is None:
        xp = get_namespace(source_points)

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
    linear_blend_skinning_fn: Any,
) -> tuple[Float[Array, "B V 3"], Float[Array, "B J 4 4"]]:
    T_world = _repose_skeleton_to_bind_pose(
        xp=xp,
        world_bind_pose_fit=world_bind_pose_fit,
        bind_pose_local=bind_pose_local,
        kinematic_fronts=kinematic_fronts,
        parents_full=parents_full,
    )
    bone = T_world @ _invert_transforms(xp, world_bind_pose_fit)
    verts = linear_blend_skinning_fn(xp, rest_shape, skin_weights, bone)
    return verts, T_world


def pose_mesh_from_oriented_pose(
    xp,
    rest_shape: Float[Array, "B V 3"],
    skin_weights: Float[Array, "V Jf"],
    world_bind_pose: Float[Array, "B Jf 4 4"],
    kinematic_fronts: list[Front],
    parents_full: list[int],
    pose_rot_full: Float[Array, "B Jf 3 3"],
    linear_blend_skinning_fn: Any,
) -> tuple[Float[Array, "B V 3"], Float[Array, "B Jf 4 4"]]:
    T_world = _pose_skeleton_from_oriented_pose(
        xp=xp,
        world_bind_pose=world_bind_pose,
        kinematic_fronts=kinematic_fronts,
        parents_full=parents_full,
        pose_rot_full=pose_rot_full,
    )
    bone = T_world @ _invert_transforms(xp, world_bind_pose)
    verts = linear_blend_skinning_fn(xp, rest_shape, skin_weights, bone)
    return verts, T_world


def apply_pose_correctives(
    data: Any,
    pose_rot_full: Float[Array, "B J 3 3"],
    *,
    xp: Any,
) -> Float[Array, "B V 3"]:
    """Compute SOMA pose correctives from oriented local rotations."""
    correctives = data.correctives
    B = pose_rot_full.shape[0]
    x = correctives.corrective_bindpose.swapaxes(-2, -1)[None] @ pose_rot_full
    x = common.set(x, (slice(None), slice(None), 0, 0), x[:, :, 0, 0] - 1.0, copy=False, xp=xp)
    x = common.set(x, (slice(None), slice(None), 1, 1), x[:, :, 1, 1] - 1.0, copy=False, xp=xp)
    feat = x[:, :, :, :2].reshape(B, -1)

    z = feat @ correctives.corrective_W1
    z = xp.maximum(z, xp.asarray(0.0, dtype=feat.dtype))

    contrib = z[:, correctives.corrective_W2_rows] * correctives.corrective_W2_values[None]
    out = common.zeros_as(z, shape=(B, data.mean_full.shape[0] * 3), xp=xp)
    out = xp.asarray(out, copy=True)
    batch_index = xp.broadcast_to(xp.arange(B)[:, None], contrib.shape)
    xp.add.at(out, (batch_index, xp.broadcast_to(correctives.corrective_W2_cols[None], contrib.shape)), contrib)
    return out.reshape(B, data.mean_full.shape[0], 3)


def identity_to_rest_vertices(
    xp,
    mean: Float[Array, "V 3"],
    shapedirs: Float[Array, "S V 3"],
    eigenvalues: Float[Array, "S"],
    identity: Float[Array, "B S"],
) -> Float[Array, "B V 3"]:
    coeffs = identity * xp.sqrt(eigenvalues)[None]
    return mean[None] + xp.einsum("bs,svc->bvc", coeffs, shapedirs)


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
    joint_positions = xp.einsum("jv,bvc->bjc", joint_regressor, rest_shape)
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
    B = world_bind_pose_fit.shape[0]
    bind_local_fit = _joint_world_to_local(xp, world_bind_pose_fit, parents_full)
    local_t = bind_local_fit[..., :3, 3]

    zeros = xp.asarray(0.0, dtype=local_t.dtype)
    local_t = common.set(local_t, (slice(None), 1, 0), zeros, copy=False, xp=xp)
    local_t = common.set(local_t, (slice(None), 1, 2), zeros, copy=False, xp=xp)

    bind_rot = xp.broadcast_to(bind_pose_local[None, :, :3, :3], (B, bind_pose_local.shape[0], 3, 3))
    T_local = _build_transform_matrix(xp, bind_rot, local_t)
    T_world = _forward_kinematics(xp, T_local, kinematic_fronts)

    y_shift = xp.amin(T_world[:, :, 1, 3], axis=1)
    return common.set(
        T_world,
        (slice(None), slice(None), 1, 3),
        T_world[:, :, 1, 3] - y_shift[:, None],
        xp=xp,
    )


def _orient_pose_rot_full(
    xp,
    pose_rot: Float[Array, "B J 3 3"],
    t_pose_world: Float[Array, "Jf 4 4"],
    parents_full: list[int],
) -> Float[Array, "B Jf 3 3"]:
    B = pose_rot.shape[0]
    root_identity = common.eye_as(pose_rot, batch_dims=(B, 1), xp=xp)
    pose_rot_full = xp.concat([root_identity, pose_rot], axis=1)
    orient = t_pose_world[:, :3, :3]
    orient_parent_T = orient[xp.asarray(parents_full)].swapaxes(-2, -1)
    return orient_parent_T[None] @ pose_rot_full @ orient[None]


def _pose_skeleton_from_oriented_pose(
    xp,
    world_bind_pose: Float[Array, "B Jf 4 4"],
    kinematic_fronts: list[Front],
    parents_full: list[int],
    pose_rot_full: Float[Array, "B Jf 3 3"],
) -> Float[Array, "B Jf 4 4"]:
    B = pose_rot_full.shape[0]
    bind_local = _joint_world_to_local(xp, world_bind_pose, parents_full)
    local_t = bind_local[..., :3, 3]
    if local_t.shape[1] > 1:
        zeros = common.zeros_as(local_t[:, 1], shape=(B, 3), xp=xp)
        local_t = common.set(local_t, (slice(None), 1), zeros, copy=False, xp=xp)

    T_local = _build_transform_matrix(xp, pose_rot_full, local_t)
    return _forward_kinematics(xp, T_local, kinematic_fronts)


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
    B, J = joint_positions.shape[:2]
    bind_rot = bind_pose_world[:, :3, :3]
    bind_pos = bind_pose_world[:, :3, 3]

    rotations = [xp.broadcast_to(bind_rot[None, 0], (B, 3, 3))]
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
            skinned_new = target_shape[:, skinned_idx, :] - joint_positions[:, joint_index : joint_index + 1, :]
            R_init = _align_vectors(
                xp,
                skinned_new,
                skinned_orig[None],
                match_warp=match_warp,
            )
        else:
            R_init = common.eye_as(bind_rot, batch_dims=(B,), xp=xp)

        child_idx = joint_children_indices_full[joint_index, : len(children)]
        pos_children_orig = bind_pos[child_idx] - bind_pos[joint_index : joint_index + 1]
        pos_children_orig = xp.einsum("bij,cj->bci", R_init, pos_children_orig)
        pos_children_new = joint_positions[:, child_idx, :] - joint_positions[:, joint_index : joint_index + 1, :]
        align_rot = _align_vectors(
            xp,
            pos_children_new,
            pos_children_orig,
            match_warp=match_warp,
        )
        R_joint = align_rot @ R_init @ bind_rot[None, joint_index]
        rotations.append(R_joint)

    R = xp.stack(rotations, axis=1)
    return _build_transform_matrix(xp, R, joint_positions)


def _align_vectors(
    xp,
    target: Float[Array, "B N 3"],
    source: Float[Array, "B N 3"],
    *,
    match_warp: bool,
) -> Float[Array, "B 3 3"]:
    if target.shape[-2] == 1:
        return _rotation_between_vectors(xp, target[:, 0], source[:, 0])

    H = xp.einsum("bni,bnj->bij", target, source)
    # SOMALayer's default warp path uses plain covariance; the alternate path adds a virtual normal.
    if not match_warp:
        p0, p1 = target[:, 0], target[:, 1]
        q0, q1 = source[:, 0], source[:, 1]
        n_target = xp.linalg.cross(p0, p1)
        n_source = xp.linalg.cross(q0, q1)
        len_target = xp.linalg.vector_norm(n_target, axis=-1, keepdims=True)
        len_source = xp.linalg.vector_norm(n_source, axis=-1, keepdims=True)
        scale_target = xp.linalg.vector_norm(p0, axis=-1, keepdims=True) / (len_target + 1e-8)
        scale_source = xp.linalg.vector_norm(q0, axis=-1, keepdims=True) / (len_source + 1e-8)
        valid = (len_target[:, 0] > 1e-9) & (len_source[:, 0] > 1e-9)
        v_target = n_target * scale_target
        v_source = n_source * scale_source
        virtual = xp.einsum("bi,bj->bij", v_target, v_source)
        H = xp.where(valid[:, None, None], H + virtual, H)
    return _kabsch(xp, H)


def _kabsch(xp, H: Float[Array, "B 3 3"]) -> Float[Array, "B 3 3"]:
    U, _, Vh = xp.linalg.svd(H)
    UVt = U @ Vh.swapaxes(-2, -1)
    det_sign = xp.where(_det3(UVt) < 0, xp.asarray(-1.0, dtype=H.dtype), xp.asarray(1.0, dtype=H.dtype))
    D = common.eye_as(H, batch_dims=(H.shape[0],), xp=xp)
    D = common.set(D, (slice(None), 2, 2), det_sign, xp=xp)
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

    antiparallel = dot[:, 0] < -1.0 + 1e-6
    basis = common.eye_as(target, batch_dims=(target.shape[0],), xp=xp)
    x_vec = basis[:, 0]
    y_vec = basis[:, 1]
    w = xp.where(xp.abs(source_unit[:, 0:1]) > 0.6, y_vec, x_vec)
    antiparallel_axis = xp.linalg.cross(source_unit, w)
    antiparallel_axis_norm = xp.linalg.vector_norm(antiparallel_axis, axis=-1, keepdims=True)
    antiparallel_axis = antiparallel_axis / xp.where(
        antiparallel_axis_norm > 1e-8,
        antiparallel_axis_norm,
        xp.ones_like(antiparallel_axis_norm),
    )
    axis = xp.where(antiparallel[:, None], antiparallel_axis, axis)

    angle = xp.atan2(cross_norm[:, 0], dot[:, 0])
    pi = xp.asarray(3.141592653589793, dtype=target.dtype)
    angle = xp.where(antiparallel, pi, angle)
    return SO3.conversions.from_axis_angle_to_rotmat(angle[..., None] * axis, xp=xp)


def _det3(M: Float[Array, "B 3 3"]) -> Float[Array, "B"]:
    return (
        M[:, 0, 0] * (M[:, 1, 1] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 1])
        - M[:, 0, 1] * (M[:, 1, 0] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 0])
        + M[:, 0, 2] * (M[:, 1, 0] * M[:, 2, 1] - M[:, 1, 1] * M[:, 2, 0])
    )


def _joint_world_to_local(xp, world: Float[Array, "B J 4 4"], parents_full: list[int]) -> Float[Array, "B J 4 4"]:
    inv = _invert_transforms(xp, world)
    return inv[:, xp.asarray(parents_full)] @ world


def linear_blend_skinning(
    xp,
    bind_shape: Float[Array, "B V 3"],
    skin_weights: Float[Array, "V J"],
    bone_transforms: Float[Array, "B J 4 4"],
) -> Float[Array, "B V 3"]:
    R = bone_transforms[..., :3, :3]
    t = bone_transforms[..., :3, 3]
    R_blend = xp.einsum("vj,bjik->bvik", skin_weights, R)
    t_blend = xp.einsum("vj,bji->bvi", skin_weights, t)
    return xp.einsum("bvik,bvk->bvi", R_blend, bind_shape) + t_blend


def _invert_transforms(xp, T: Float[Array, "B J 4 4"]) -> Float[Array, "B J 4 4"]:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_T = R.swapaxes(-2, -1)
    t_inv = -xp.squeeze(R_T @ t[..., None], axis=-1)
    return _build_transform_matrix(xp, R_T, t_inv)


def _build_transform_matrix(
    xp,
    R: Float[Array, "... 3 3"],
    t: Float[Array, "... 3"],
) -> Float[Array, "... 4 4"]:
    T = common.zeros_as(R, shape=(*R.shape[:-2], 4, 4), xp=xp)
    T = common.set(T, (..., slice(None, 3), slice(None, 3)), R, xp=xp)
    T = common.set(T, (..., slice(None, 3), 3), t, xp=xp)
    T = common.set(T, (..., 3, 3), xp.asarray(1.0, dtype=R.dtype), xp=xp)
    return T


def _forward_kinematics(xp, T_local: Float[Array, "B J 4 4"], fronts: list[Front]) -> Float[Array, "B J 4 4"]:
    J = T_local.shape[1]
    world: list[Float[Array, "B 4 4"] | None] = [None] * J
    for joints, parents in fronts:
        if parents[0] < 0:
            for joint in joints:
                world[joint] = T_local[:, joint]
            continue
        for joint, parent in zip(joints, parents):
            world[joint] = world[parent] @ T_local[:, joint]
    return xp.stack(world, axis=1)


def _apply_global_transform_vertices(
    xp,
    vertices: Float[Array, "B V 3"],
    rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None,
    translation: Float[Array, "B 3"] | None,
    rotation_type: RotationType,
) -> Float[Array, "B V 3"]:
    if rotation is not None:
        R = SO3.convert(rotation, src=rotation_type, dst="rotmat", xp=xp)
        vertices = xp.einsum("bij,bvj->bvi", R, vertices)
    if translation is not None:
        vertices = vertices + translation[:, None]
    return vertices


def _apply_global_transform_transforms(
    xp,
    transforms: Float[Array, "B J 4 4"],
    rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None,
    translation: Float[Array, "B 3"] | None,
    rotation_type: RotationType,
) -> Float[Array, "B J 4 4"]:
    if rotation is None and translation is None:
        return transforms

    B = transforms.shape[0]
    global_T = common.zeros_as(transforms, shape=(B, 4, 4), xp=xp)
    global_T = common.set(global_T, (slice(None), 3, 3), xp.asarray(1.0, dtype=transforms.dtype), xp=xp)
    if rotation is not None:
        R = SO3.convert(rotation, src=rotation_type, dst="rotmat", xp=xp)
        global_T = common.set(global_T, (slice(None), slice(None, 3), slice(None, 3)), R, xp=xp)
    else:
        eye = common.eye_as(transforms, batch_dims=(B,), xp=xp)
        global_T = common.set(global_T, (slice(None), slice(None, 3), slice(None, 3)), eye[:, :3, :3], xp=xp)
    if translation is not None:
        global_T = common.set(global_T, (slice(None), slice(None, 3), 3), translation, xp=xp)
    return xp.einsum("bij,bnjk->bnik", global_T, transforms)
