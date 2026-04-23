"""Backend-agnostic SOMA computation using array_api_compat."""

from __future__ import annotations

from typing import Any

from array_api_compat import get_namespace
from jaxtyping import Float, Int
from nanomanifold import SO3

from .. import common
from ..rotations import RotationType

Array = Any
Front = tuple[list[int], list[int]]


def forward_vertices(
    mean_full: Float[Array, "Vf 3"],
    mean_active: Float[Array, "Va 3"],
    shapedirs_full: Float[Array, "S Vf 3"],
    shapedirs_active: Float[Array, "S Va 3"],
    eigenvalues: Float[Array, "S"],
    bind_shape_full: Float[Array, "Vf 3"],
    skin_weights_active: Float[Array, "Va Jf"],
    bind_pose_world: Float[Array, "Jf 4 4"],
    bind_pose_local: Float[Array, "Jf 4 4"],
    t_pose_world: Float[Array, "Jf 4 4"],
    joint_regressor: Float[Array, "Jf Vf"],
    joint_children_full: list[list[int]],
    skinned_vertex_indices_full: list[list[int]],
    kinematic_fronts_full: list[Front],
    parents_full: list[int],
    shape: Float[Array, "B|1 S"] | None,
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    corrective_bindpose: Float[Array, "Jf 3 3"],
    corrective_W1: Float[Array, "D K"],
    corrective_W2_rows: Int[Array, "NNZ"],
    corrective_W2_cols: Int[Array, "NNZ"],
    corrective_W2_values: Float[Array, "NNZ"],
    rest_shape_full: Float[Array, "B|1 Vf 3"] | None = None,
    rest_shape_active: Float[Array, "B|1 Va 3"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    vertex_map: Int[Array, "Va"] | None = None,
    corrective_use_tanh: bool = True,
    apply_correctives: bool = True,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3] in meters."""
    if xp is None:
        xp = get_namespace(shape if shape is not None else rest_shape_full)

    pose_rot = SO3.convert(pose, src=rotation_type, dst="rotmat", xp=xp)
    B = pose_rot.shape[0]
    pose_rot_full = _orient_pose_rot_full(xp, pose_rot, t_pose_world, parents_full)
    rest_shape_full, rest_shape_active, world_bind_pose_fit = _prepare_rest_shapes(
        xp=xp,
        batch_size=B,
        mean_full=mean_full,
        mean_active=mean_active,
        shapedirs_full=shapedirs_full,
        shapedirs_active=shapedirs_active,
        eigenvalues=eigenvalues,
        bind_shape=bind_shape_full,
        bind_pose_world=bind_pose_world,
        joint_regressor=joint_regressor,
        joint_children_full=joint_children_full,
        skinned_vertex_indices_full=skinned_vertex_indices_full,
        parents_full=parents_full,
        shape=shape,
        rest_shape_full=rest_shape_full,
        rest_shape_active=rest_shape_active,
    )

    if apply_correctives:
        rest_shape_active, world_bind_pose = _repose_to_bind_pose(
            xp=xp,
            rest_shape=rest_shape_active,
            skin_weights=skin_weights_active,
            world_bind_pose_fit=world_bind_pose_fit,
            bind_pose_local=bind_pose_local,
            kinematic_fronts=kinematic_fronts_full,
            parents_full=parents_full,
        )
        corrective_offsets = apply_pose_correctives(
            pose_rot_full=pose_rot_full,
            bindpose=corrective_bindpose,
            W1=corrective_W1,
            W2_rows=corrective_W2_rows,
            W2_cols=corrective_W2_cols,
            W2_values=corrective_W2_values,
            num_vertices=mean_full.shape[0],
            use_tanh=corrective_use_tanh,
            xp=xp,
        )
        if vertex_map is not None:
            corrective_offsets = corrective_offsets[:, vertex_map]
        rest_shape_active = rest_shape_active + corrective_offsets
    else:
        world_bind_pose = world_bind_pose_fit

    verts_cm, _ = _pose_mesh_from_oriented_pose(
        xp=xp,
        rest_shape=rest_shape_active,
        skin_weights=skin_weights_active,
        world_bind_pose=world_bind_pose,
        kinematic_fronts=kinematic_fronts_full,
        parents_full=parents_full,
        pose_rot_full=pose_rot_full,
    )

    verts = verts_cm * 0.01
    verts = _apply_global_transform_vertices(xp, verts, global_rotation, global_translation, rotation_type)
    if vertex_indices is not None:
        verts = verts[:, xp.asarray(vertex_indices)]
    return verts


def forward_skeleton(
    mean_full: Float[Array, "V 3"],
    shapedirs_full: Float[Array, "S V 3"],
    eigenvalues: Float[Array, "S"],
    bind_shape_full: Float[Array, "V 3"],
    skin_weights_full: Float[Array, "V Jf"],
    bind_pose_world: Float[Array, "Jf 4 4"],
    bind_pose_local: Float[Array, "Jf 4 4"],
    t_pose_world: Float[Array, "Jf 4 4"],
    joint_regressor: Float[Array, "Jf V"],
    joint_children_full: list[list[int]],
    skinned_vertex_indices_full: list[list[int]],
    kinematic_fronts_full: list[Front],
    parents_full: list[int],
    shape: Float[Array, "B|1 S"] | None,
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    rest_shape_full: Float[Array, "B|1 V 3"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    apply_correctives: bool = True,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
    """Compute skeleton transforms [B, J, 4, 4] in meters."""
    if xp is None:
        xp = get_namespace(shape if shape is not None else rest_shape_full)

    pose_rot = SO3.convert(pose, src=rotation_type, dst="rotmat", xp=xp)
    B = pose_rot.shape[0]
    rest_shape_full, world_bind_pose = _prepare_identity_from_inputs(
        xp=xp,
        batch_size=B,
        mean=mean_full,
        shapedirs=shapedirs_full,
        eigenvalues=eigenvalues,
        bind_shape=bind_shape_full,
        bind_pose_world=bind_pose_world,
        joint_regressor=joint_regressor,
        joint_children_full=joint_children_full,
        skinned_vertex_indices_full=skinned_vertex_indices_full,
        parents_full=parents_full,
        shape=shape,
        rest_shape=rest_shape_full,
    )
    if apply_correctives:
        rest_shape_full, world_bind_pose = _repose_to_bind_pose(
            xp=xp,
            rest_shape=rest_shape_full,
            skin_weights=skin_weights_full,
            world_bind_pose_fit=world_bind_pose,
            bind_pose_local=bind_pose_local,
            kinematic_fronts=kinematic_fronts_full,
            parents_full=parents_full,
        )
    pose_rot_full = _orient_pose_rot_full(xp, pose_rot, t_pose_world, parents_full)
    _, T_world_cm = _pose_mesh_from_oriented_pose(
        xp=xp,
        rest_shape=rest_shape_full,
        skin_weights=skin_weights_full,
        world_bind_pose=world_bind_pose,
        kinematic_fronts=kinematic_fronts_full,
        parents_full=parents_full,
        pose_rot_full=pose_rot_full,
    )

    T_world = common.set(T_world_cm, (..., slice(None, 3), 3), T_world_cm[..., :3, 3] * 0.01, xp=xp)
    T_world = _apply_global_transform_transforms(xp, T_world, global_rotation, global_translation, rotation_type)
    public = T_world[:, 1:]
    if joint_indices is not None:
        public = public[:, xp.asarray(joint_indices)]
    return public


def transfer_identity_rest_shape(
    source_shape: Float[Array, "B Vs 3"],
    source_tetrahedra: Int[Array, "Fs 4"],
    face_ids: Int[Array, "Vt"],
    bary_coords: Float[Array, "Vt 4"],
    unknown_ids: Int[Array, "U"],
    anchor_ids: Int[Array, "A"],
    solve_matrix: Float[Array, "U U"],
    anchor_matrix: Float[Array, "U A"],
    rhs_base: Float[Array, "U 3"],
    *,
    xp: Any = None,
) -> Float[Array, "B Vt 3"]:
    if xp is None:
        xp = get_namespace(source_shape)

    tetra_faces = source_tetrahedra[:, :3]
    f0 = source_shape[:, tetra_faces[:, 0]]
    f1 = source_shape[:, tetra_faces[:, 1]]
    f2 = source_shape[:, tetra_faces[:, 2]]
    fabricated = f0 + xp.linalg.cross(f1 - f0, f2 - f0)
    source_shape_tet = xp.concat([source_shape, fabricated], axis=1)

    tet_indices = source_tetrahedra[face_ids]
    v0 = source_shape_tet[:, tet_indices[:, 0]]
    v1 = source_shape_tet[:, tet_indices[:, 1]]
    v2 = source_shape_tet[:, tet_indices[:, 2]]
    v3 = source_shape_tet[:, tet_indices[:, 3]]
    bc = bary_coords[None]
    target_shape = v0 * bc[..., 0:1] + v1 * bc[..., 1:2] + v2 * bc[..., 2:3] + v3 * bc[..., 3:4]

    if unknown_ids.shape[0] == 0:
        return target_shape

    B = target_shape.shape[0]
    num_unknown = unknown_ids.shape[0]
    num_anchor = anchor_ids.shape[0]
    anchor_vertices = target_shape[:, anchor_ids]
    anchor_vertices = xp.reshape(anchor_vertices.swapaxes(0, 1), (num_anchor, B * 3))
    rhs = xp.broadcast_to(rhs_base[:, None, :], (num_unknown, B, 3)).reshape(num_unknown, B * 3)
    rhs = rhs - anchor_matrix @ anchor_vertices
    unknown_vertices = xp.linalg.solve(solve_matrix, rhs)
    unknown_vertices = xp.reshape(unknown_vertices, (num_unknown, B, 3)).swapaxes(0, 1)
    return common.set(target_shape, (slice(None), unknown_ids), unknown_vertices, xp=xp)


def linear_identity_shape(
    mean: Float[Array, "V 3"],
    shapedirs: Float[Array, "V 3 S"] | Float[Array, "V 3 I"],
    identity: Float[Array, "B I"],
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    if xp is None:
        xp = get_namespace(identity)
    identity_dim = identity.shape[1]
    return mean[None] + xp.einsum("bi,vci->bvc", identity, shapedirs[..., :identity_dim])


def mhr_identity_shape(
    model: Any,
    identity: Float[Array, "B I"],
    scale_params: Float[Array, "B K"] | None,
    num_scale_params: int,
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    if xp is None:
        xp = get_namespace(identity)

    batch_size = identity.shape[0]
    if scale_params is None:
        scale_params = common.zeros_as(identity, shape=(batch_size, num_scale_params), xp=xp)
    zero_pose = common.zeros_as(identity, shape=(batch_size, model.pose_dim), xp=xp)
    zero_pose = common.set(zero_pose, (slice(None), slice(-num_scale_params, None)), scale_params, xp=xp)
    expression = common.zeros_as(identity, shape=(batch_size, model.EXPR_DIM), xp=xp)
    return model.forward_vertices(shape=identity, pose=zero_pose, expression=expression)


def resolve_identity_inputs(
    model_type: str,
    shape: Float[Array, "B|1 S"] | None,
    identity: Float[Array, "B|1 I"] | None,
    scale_params: Float[Array, "B|1 K"] | None,
    *,
    batch_size: int,
    identity_dim: int,
    num_scale_params: int | None,
    ref: Array,
    xp: Any = None,
) -> tuple[Float[Array, "B|1 S"] | None, Float[Array, "B I"] | None, Float[Array, "B K"] | None]:
    if xp is None:
        xp = get_namespace(ref)

    if model_type == "soma":
        if identity is not None:
            if shape is not None:
                raise ValueError("Pass either shape or identity for SOMA model_type='soma', not both.")
            shape = identity
        if shape is None:
            raise ValueError("SOMA model_type='soma' requires shape or identity coefficients.")
        return shape, None, None

    if shape is not None:
        raise ValueError("shape is only supported for SOMA model_type='soma'. Use identity for other backends.")

    if identity is None:
        identity = common.zeros_as(ref, shape=(1, identity_dim), xp=xp)
    if identity.shape[0] == 1 and batch_size > 1:
        identity = xp.broadcast_to(identity, (batch_size, identity.shape[-1]))

    if num_scale_params is None:
        if scale_params is not None:
            raise ValueError("scale_params is only supported for SOMA model_type='mhr'.")
        return None, identity, scale_params

    if scale_params is None:
        scale_params = common.zeros_as(ref, shape=(1, num_scale_params), xp=xp)
    if scale_params.shape[0] == 1 and batch_size > 1:
        scale_params = xp.broadcast_to(scale_params, (batch_size, scale_params.shape[-1]))
    return None, identity, scale_params


def apply_pose_correctives(
    pose_rot_full: Float[Array, "B J 3 3"],
    bindpose: Float[Array, "J 3 3"],
    W1: Float[Array, "D K"],
    W2_rows: Int[Array, "NNZ"],
    W2_cols: Int[Array, "NNZ"],
    W2_values: Float[Array, "NNZ"],
    num_vertices: int,
    use_tanh: bool,
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute SOMA pose correctives from oriented local rotations."""
    if xp is None:
        xp = get_namespace(pose_rot_full)

    B = pose_rot_full.shape[0]
    x = bindpose.swapaxes(-2, -1)[None] @ pose_rot_full
    x = common.set(x, (slice(None), slice(None), 0, 0), x[:, :, 0, 0] - 1.0, copy=False, xp=xp)
    x = common.set(x, (slice(None), slice(None), 1, 1), x[:, :, 1, 1] - 1.0, copy=False, xp=xp)
    feat = x[:, :, :, :2].reshape(B, -1)

    z = feat @ W1
    z = xp.maximum(z, xp.asarray(0.0, dtype=feat.dtype))
    if use_tanh:
        z = xp.tanh(z)

    contrib = z[:, W2_rows] * W2_values[None]
    out = common.zeros_as(z, shape=(B, num_vertices * 3), xp=xp)
    if "torch" in xp.__name__:
        out = out.clone()
        index = xp.broadcast_to(W2_cols[None], contrib.shape)
        out = out.scatter_add(1, index, contrib)
    elif "jax" in xp.__name__:
        out = out.at[:, W2_cols].add(contrib)
    else:
        out = xp.asarray(out, copy=True)
        batch_index = xp.broadcast_to(xp.arange(B)[:, None], contrib.shape)
        xp.add.at(out, (batch_index, xp.broadcast_to(W2_cols[None], contrib.shape)), contrib)
    return out.reshape(B, num_vertices, 3)


def _broadcast_shape(shape: Array, *, batch_size: int, xp: Any) -> Array:
    if shape.shape[0] == 1 and batch_size > 1:
        return xp.broadcast_to(shape, (batch_size, shape.shape[-1]))
    return shape


def _broadcast_rest_shape(rest_shape: Array, *, batch_size: int, xp: Any) -> Array:
    if rest_shape.shape[0] == 1 and batch_size > 1:
        return xp.broadcast_to(rest_shape, (batch_size, *rest_shape.shape[1:]))
    return rest_shape


def _shape_to_rest_vertices(
    xp,
    mean: Float[Array, "V 3"],
    shapedirs: Float[Array, "S V 3"],
    eigenvalues: Float[Array, "S"],
    shape: Float[Array, "B S"],
) -> Float[Array, "B V 3"]:
    coeffs = shape * xp.sqrt(eigenvalues)[None]
    return mean[None] + xp.einsum("bs,svc->bvc", coeffs, shapedirs)


def _fit_rest_shape_to_bind_pose(
    xp,
    bind_shape: Float[Array, "V 3"],
    bind_pose_world: Float[Array, "J 4 4"],
    joint_regressor: Float[Array, "J V"],
    joint_children_full: list[list[int]],
    skinned_vertex_indices_full: list[list[int]],
    parents_full: list[int],
    rest_shape: Float[Array, "B V 3"],
) -> tuple[Float[Array, "B V 3"], Float[Array, "B J 4 4"]]:
    joint_positions = xp.einsum("jv,bvc->bjc", joint_regressor, rest_shape)
    world_bind_pose = _fit_joint_rotations(
        xp=xp,
        bind_shape=bind_shape,
        bind_pose_world=bind_pose_world,
        joint_children_full=joint_children_full,
        skinned_vertex_indices_full=skinned_vertex_indices_full,
        parents_full=parents_full,
        joint_positions=joint_positions,
        target_shape=rest_shape,
    )
    return rest_shape, world_bind_pose


def _prepare_identity_from_inputs(
    xp,
    batch_size: int,
    mean: Float[Array, "V 3"],
    shapedirs: Float[Array, "S V 3"],
    eigenvalues: Float[Array, "S"],
    bind_shape: Float[Array, "V 3"],
    bind_pose_world: Float[Array, "J 4 4"],
    joint_regressor: Float[Array, "J V"],
    joint_children_full: list[list[int]],
    skinned_vertex_indices_full: list[list[int]],
    parents_full: list[int],
    shape: Float[Array, "B|1 S"] | None,
    rest_shape: Float[Array, "B|1 V 3"] | None,
) -> tuple[Float[Array, "B V 3"], Float[Array, "B J 4 4"]]:
    if rest_shape is not None:
        rest_shape = _broadcast_rest_shape(rest_shape, batch_size=batch_size, xp=xp)
    else:
        if shape is None:
            raise ValueError("SOMA forward pass requires either shape or a precomputed rest_shape.")
        shape = _broadcast_shape(shape, batch_size=batch_size, xp=xp)
        rest_shape = _shape_to_rest_vertices(xp, mean, shapedirs, eigenvalues, shape)
    return _fit_rest_shape_to_bind_pose(
        xp=xp,
        bind_shape=bind_shape,
        bind_pose_world=bind_pose_world,
        joint_regressor=joint_regressor,
        joint_children_full=joint_children_full,
        skinned_vertex_indices_full=skinned_vertex_indices_full,
        parents_full=parents_full,
        rest_shape=rest_shape,
    )


def _prepare_rest_shapes(
    xp,
    batch_size: int,
    mean_full: Float[Array, "Vf 3"],
    mean_active: Float[Array, "Va 3"],
    shapedirs_full: Float[Array, "S Vf 3"],
    shapedirs_active: Float[Array, "S Va 3"],
    eigenvalues: Float[Array, "S"],
    bind_shape: Float[Array, "Vf 3"],
    bind_pose_world: Float[Array, "Jf 4 4"],
    joint_regressor: Float[Array, "Jf Vf"],
    joint_children_full: list[list[int]],
    skinned_vertex_indices_full: list[list[int]],
    parents_full: list[int],
    shape: Float[Array, "B|1 S"] | None,
    rest_shape_full: Float[Array, "B|1 Vf 3"] | None,
    rest_shape_active: Float[Array, "B|1 Va 3"] | None,
) -> tuple[Float[Array, "B Vf 3"], Float[Array, "B Va 3"], Float[Array, "B Jf 4 4"]]:
    full_shape, world_bind_pose = _prepare_identity_from_inputs(
        xp=xp,
        batch_size=batch_size,
        mean=mean_full,
        shapedirs=shapedirs_full,
        eigenvalues=eigenvalues,
        bind_shape=bind_shape,
        bind_pose_world=bind_pose_world,
        joint_regressor=joint_regressor,
        joint_children_full=joint_children_full,
        skinned_vertex_indices_full=skinned_vertex_indices_full,
        parents_full=parents_full,
        shape=shape,
        rest_shape=rest_shape_full,
    )
    if rest_shape_active is not None:
        active_shape = _broadcast_rest_shape(rest_shape_active, batch_size=batch_size, xp=xp)
    else:
        if shape is None:
            active_shape = full_shape
        else:
            shape = _broadcast_shape(shape, batch_size=batch_size, xp=xp)
            active_shape = _shape_to_rest_vertices(xp, mean_active, shapedirs_active, eigenvalues, shape)
    return full_shape, active_shape, world_bind_pose


def _repose_to_bind_pose(
    xp,
    rest_shape: Float[Array, "B V 3"],
    skin_weights: Float[Array, "V J"],
    world_bind_pose_fit: Float[Array, "B J 4 4"],
    bind_pose_local: Float[Array, "J 4 4"],
    kinematic_fronts: list[Front],
    parents_full: list[int],
) -> tuple[Float[Array, "B V 3"], Float[Array, "B J 4 4"]]:
    """Repose an identity-specific rest shape into SOMA's corrective bind pose."""
    B = rest_shape.shape[0]
    bind_local_fit = _joint_world_to_local(xp, world_bind_pose_fit, parents_full)
    local_t = bind_local_fit[..., :3, 3]

    zeros = xp.asarray(0.0, dtype=local_t.dtype)
    local_t = common.set(local_t, (slice(None), 1, 0), zeros, copy=False, xp=xp)
    local_t = common.set(local_t, (slice(None), 1, 2), zeros, copy=False, xp=xp)

    bind_rot = xp.broadcast_to(bind_pose_local[None, :, :3, :3], (B, bind_pose_local.shape[0], 3, 3))
    T_local = _build_transform_matrix(xp, bind_rot, local_t)
    T_world = _forward_kinematics(xp, T_local, kinematic_fronts)

    y_shift = xp.amin(T_world[:, :, 1, 3], axis=1)
    T_world = common.set(
        T_world,
        (slice(None), slice(None), 1, 3),
        T_world[:, :, 1, 3] - y_shift[:, None],
        xp=xp,
    )

    bone = T_world @ _invert_transforms(xp, world_bind_pose_fit)
    verts = _linear_blend_skinning(xp, rest_shape, skin_weights, bone)
    return verts, T_world


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


def _pose_mesh_from_oriented_pose(
    xp,
    rest_shape: Float[Array, "B V 3"],
    skin_weights: Float[Array, "V Jf"],
    world_bind_pose: Float[Array, "B Jf 4 4"],
    kinematic_fronts: list[Front],
    parents_full: list[int],
    pose_rot_full: Float[Array, "B Jf 3 3"],
) -> tuple[Float[Array, "B V 3"], Float[Array, "B Jf 4 4"]]:
    B = pose_rot_full.shape[0]
    bind_local = _joint_world_to_local(xp, world_bind_pose, parents_full)
    local_t = bind_local[..., :3, 3]
    if local_t.shape[1] > 1:
        zeros = common.zeros_as(local_t[:, 1], shape=(B, 3), xp=xp)
        local_t = common.set(local_t, (slice(None), 1), zeros, copy=False, xp=xp)

    T_local = _build_transform_matrix(xp, pose_rot_full, local_t)
    T_world = _forward_kinematics(xp, T_local, kinematic_fronts)
    bone = T_world @ _invert_transforms(xp, world_bind_pose)
    verts = _linear_blend_skinning(xp, rest_shape, skin_weights, bone)
    return verts, T_world


def _fit_joint_rotations(
    xp,
    bind_shape: Float[Array, "V 3"],
    bind_pose_world: Float[Array, "J 4 4"],
    joint_children_full: list[list[int]],
    skinned_vertex_indices_full: list[list[int]],
    parents_full: list[int],
    joint_positions: Float[Array, "B J 3"],
    target_shape: Float[Array, "B V 3"],
) -> Float[Array, "B J 4 4"]:
    B, J = joint_positions.shape[:2]
    bind_rot = bind_pose_world[:, :3, :3]
    bind_pos = bind_pose_world[:, :3, 3]

    R = common.zeros_as(bind_rot, shape=(B, J, 3, 3), xp=xp)
    R = common.set(R, (..., slice(None), slice(None)), xp.broadcast_to(bind_rot[None], (B, J, 3, 3)), xp=xp)
    for joint_index in range(1, J):
        children = joint_children_full[joint_index]
        if not children:
            parent_index = parents_full[joint_index]
            R = common.set(R, (slice(None), joint_index), R[:, parent_index], copy=False, xp=xp)
            continue

        skinned_vids = skinned_vertex_indices_full[joint_index]
        if skinned_vids:
            skinned_idx = xp.asarray(skinned_vids)
            skinned_orig = bind_shape[skinned_idx] - bind_pos[joint_index]
            skinned_new = target_shape[:, skinned_idx, :] - joint_positions[:, joint_index : joint_index + 1, :]
            R_init = _align_vectors(xp, skinned_new, skinned_orig[None])
        else:
            R_init = common.eye_as(bind_rot, batch_dims=(B,), xp=xp)

        child_idx = xp.asarray(children)
        pos_children_orig = bind_pos[child_idx] - bind_pos[joint_index : joint_index + 1]
        pos_children_orig = xp.einsum("bij,cj->bci", R_init, pos_children_orig)
        pos_children_new = joint_positions[:, child_idx, :] - joint_positions[:, joint_index : joint_index + 1, :]
        align_rot = _align_vectors(xp, pos_children_new, pos_children_orig)
        R_joint = align_rot @ R_init @ R[:, joint_index]
        R = common.set(R, (slice(None), joint_index), R_joint, copy=False, xp=xp)

    return _build_transform_matrix(xp, R, joint_positions)


def _align_vectors(
    xp,
    target: Float[Array, "B N 3"],
    source: Float[Array, "B N 3"],
) -> Float[Array, "B 3 3"]:
    if target.shape[-2] == 1:
        return _rodrigues_rotation(xp, target[:, 0], source[:, 0])

    H = xp.einsum("bni,bnj->bij", target, source)
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


def _rodrigues_rotation(
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
    v = xp.linalg.cross(source_unit, target_unit)
    K = _skew(xp, v)
    I = common.eye_as(target, batch_dims=(target.shape[0],), xp=xp)
    denom = 1.0 + dot[..., None]
    R = I + K + (K @ K) / xp.where(denom > 1e-8, denom, xp.ones_like(denom))

    antiparallel = dot[:, 0] < -1.0 + 1e-6
    if bool(xp.any(antiparallel)):
        x_vec = common.zeros_as(target, shape=target.shape, xp=xp)
        x_vec = common.set(x_vec, (slice(None), 0), xp.asarray(1.0, dtype=target.dtype), xp=xp)
        y_vec = common.zeros_as(target, shape=target.shape, xp=xp)
        y_vec = common.set(y_vec, (slice(None), 1), xp.asarray(1.0, dtype=target.dtype), xp=xp)
        w = xp.where(xp.abs(source_unit[:, 0:1]) > 0.6, y_vec, x_vec)
        axis = xp.linalg.cross(source_unit, w)
        axis = axis / xp.where(
            xp.linalg.vector_norm(axis, axis=-1, keepdims=True) > 1e-8,
            xp.linalg.vector_norm(axis, axis=-1, keepdims=True),
            xp.ones_like(axis[:, :1]),
        )
        uuT = axis[..., :, None] * axis[..., None, :]
        R_180 = 2.0 * uuT - I
        R = xp.where(antiparallel[:, None, None], R_180, R)
    return R


def _skew(xp, v: Float[Array, "B 3"]) -> Float[Array, "B 3 3"]:
    z = xp.zeros_like(v[:, :1])
    row0 = xp.concat([z, -v[:, 2:3], v[:, 1:2]], axis=-1)
    row1 = xp.concat([v[:, 2:3], z, -v[:, 0:1]], axis=-1)
    row2 = xp.concat([-v[:, 1:2], v[:, 0:1], z], axis=-1)
    return xp.stack([row0, row1, row2], axis=-2)


def _det3(M: Float[Array, "B 3 3"]) -> Float[Array, "B"]:
    return (
        M[:, 0, 0] * (M[:, 1, 1] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 1])
        - M[:, 0, 1] * (M[:, 1, 0] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 0])
        + M[:, 0, 2] * (M[:, 1, 0] * M[:, 2, 1] - M[:, 1, 1] * M[:, 2, 0])
    )


def _joint_world_to_local(xp, world: Float[Array, "B J 4 4"], parents_full: list[int]) -> Float[Array, "B J 4 4"]:
    inv = _invert_transforms(xp, world)
    return inv[:, xp.asarray(parents_full)] @ world


def _linear_blend_skinning(
    xp,
    bind_shape: Float[Array, "B V 3"],
    skin_weights: Float[Array, "V J"],
    bone_transforms: Float[Array, "B J 4 4"],
) -> Float[Array, "B V 3"]:
    R = bone_transforms[..., :3, :3]
    t = bone_transforms[..., :3, 3]
    transformed = xp.einsum("bjik,bvk->bjvi", R, bind_shape) + t[:, :, None, :]
    return xp.einsum("vj,bjvi->bvi", skin_weights, transformed)


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
