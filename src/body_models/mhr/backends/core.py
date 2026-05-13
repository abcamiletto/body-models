"""Backend-agnostic MHR computation."""

import math
from typing import Any

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import common
from body_models.common import get_namespace

Array = Any  # Generic array type (numpy, torch, jax)
Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).

_LN2 = math.log(2)


def apply_pose_correctives(
    joint_params: Float[Array, "B J 7"],
    W1: Float[Array, "3000 750"],
    W2: Float[Array, "V*3 3000"],
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    if xp is None:
        xp = get_namespace(joint_params)

    V = W2.shape[0] // 3
    dtype = joint_params.dtype

    euler = joint_params[..., 2:, 3:6]
    rot = SO3.conversions.from_euler_to_rotmat(euler, convention="xyz", xp=xp)
    feat = xp.concat([rot[..., 0], rot[..., 1]], axis=-1)
    feat = common.set(feat, (..., 0), feat[..., 0] - 1.0, copy=False, xp=xp)
    feat = common.set(feat, (..., 4), feat[..., 4] - 1.0, copy=False, xp=xp)

    batch_shape = feat.shape[:-2]
    feat_flat = feat.reshape(*batch_shape, -1)
    h = feat_flat @ W1.T
    h = xp.maximum(h, xp.asarray(0.0, dtype=dtype))
    out = h @ W2.T

    return out.reshape(*batch_shape, V, 3)


def forward_vertices(
    base_vertices: Float[Array, "V 3"],
    blendshape_dirs: Float[Array, "117 V 3"],
    skin_weights: Float[Array, "V K"],
    skin_indices: Int[Array, "V K"],
    joint_offsets: Float[Array, "J 3"],
    joint_pre_rotations: Float[Array, "J 4"],
    parameter_transform: Float[Array, "D N"],
    bind_inv_linear: Float[Array, "J 3 3"],
    bind_inv_translation: Float[Array, "J 3"],
    corrective_W1: Float[Array, "3000 750"],
    corrective_W2: Float[Array, "V*3 3000"],
    kinematic_fronts: list[Front],
    num_joints: int,
    shape_dim: int,
    expr_dim: int,
    shape: Float[Array, "B 45"],
    pose: Float[Array, "B 204"],
    expression: Float[Array, "B 72"] | None = None,
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3] in meters."""
    assert shape.ndim >= 1 and shape.shape[-1] == shape_dim
    assert pose.ndim >= 1 and pose.shape[-1] == 204
    assert expression is None or (expression.ndim >= 1 and expression.shape[-1] == expr_dim)
    assert global_rotation is None or (global_rotation.ndim >= 1 and global_rotation.shape[-1] == 3)
    assert global_translation is None or (global_translation.ndim >= 1 and global_translation.shape[-1] == 3)

    if xp is None:
        xp = get_namespace(shape)
    batch_shape = tuple(pose.shape[:-1])
    shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
    assert expression is None or tuple(expression.shape[:-1]) == batch_shape

    if vertex_indices is not None:
        vertex_indices = xp.asarray(vertex_indices)
        base_vertices = base_vertices[vertex_indices]
        blendshape_dirs = blendshape_dirs[:, vertex_indices]
        skin_weights = skin_weights[vertex_indices]
        skin_indices = skin_indices[vertex_indices]
        corrective_W2 = corrective_W2.reshape(-1, 3, corrective_W2.shape[-1])[vertex_indices].reshape(
            -1, corrective_W2.shape[-1]
        )

    if expression is None:
        expression = common.zeros_as(shape, shape=(*batch_shape, expr_dim), xp=xp)

    coeffs = xp.concat([shape, expression], axis=-1)

    t_g, r_g, s_g, j_p = _forward_skeleton_core(
        xp=xp,
        pose=pose,
        joint_offsets=joint_offsets,
        joint_pre_rotations=joint_pre_rotations,
        parameter_transform=parameter_transform,
        kinematic_fronts=kinematic_fronts,
        num_joints=num_joints,
        shape_dim=shape_dim,
    )

    v_t = base_vertices + xp.einsum("...i,ivk->...vk", coeffs, blendshape_dirs)
    v_t = v_t + apply_pose_correctives(j_p, corrective_W1, corrective_W2, xp=xp)

    lin_g = r_g * s_g[..., None]
    lin = xp.einsum("...jik,jkl->...jil", lin_g, bind_inv_linear)
    t = xp.einsum("...jik,jk->...ji", lin_g, bind_inv_translation) + t_g

    lin = _gather_joint_matrices(lin, skin_indices)
    t = _gather_joint_vectors(t, skin_indices)

    v_transformed = xp.einsum("...vkij,...vj->...vki", lin, v_t) + t
    verts = xp.sum(v_transformed * skin_weights[:, :, None], axis=2)
    verts = verts * 0.01

    if global_rotation is not None:
        R = SO3.conversions.from_axis_angle_to_rotmat(global_rotation, xp=xp)
        verts = xp.einsum("...ij,...vj->...vi", R, verts)
    if global_translation is not None:
        verts = verts + global_translation[..., None, :]

    return verts


def forward_skeleton(
    joint_offsets: Float[Array, "J 3"],
    joint_pre_rotations: Float[Array, "J 4"],
    parameter_transform: Float[Array, "D N"],
    kinematic_fronts: list[Front],
    num_joints: int,
    shape_dim: int,
    pose: Float[Array, "B 204"],
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    *,
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
    """Compute skeleton transforms [B, J, 4, 4] in meters."""
    assert pose.ndim >= 1 and pose.shape[-1] == 204
    assert global_rotation is None or (global_rotation.ndim >= 1 and global_rotation.shape[-1] == 3)
    assert global_translation is None or (global_translation.ndim >= 1 and global_translation.shape[-1] == 3)

    if xp is None:
        xp = get_namespace(pose)
    batch_shape = tuple(pose.shape[:-1])
    active_fronts = kinematic_fronts
    if joint_indices is not None:
        joint_indices = [int(joint) for joint in joint_indices]
        if any(joint < 0 or joint >= num_joints for joint in joint_indices):
            raise IndexError(f"joint_indices must be in [0, {num_joints})")

        parents = [-1] * num_joints
        for joints, joint_parents in kinematic_fronts:
            for joint, parent in zip(joints, joint_parents):
                parents[joint] = parent

        active_joints = set()
        for joint in joint_indices:
            cur = joint
            while cur >= 0 and cur not in active_joints:
                active_joints.add(cur)
                cur = parents[cur]

        active_fronts = []
        for joints, joint_parents in kinematic_fronts:
            pairs = [(joint, parent) for joint, parent in zip(joints, joint_parents) if joint in active_joints]
            if pairs:
                active_fronts.append(([joint for joint, _ in pairs], [parent for _, parent in pairs]))

    t_g, r_g, s_g, _ = _forward_skeleton_core(
        xp=xp,
        pose=pose,
        joint_offsets=joint_offsets,
        joint_pre_rotations=joint_pre_rotations,
        parameter_transform=parameter_transform,
        kinematic_fronts=active_fronts,
        num_joints=num_joints,
        shape_dim=shape_dim,
        joint_indices=joint_indices,
    )

    T = _trs_to_transforms(xp, t_g * 0.01, r_g, s_g)

    if global_rotation is not None or global_translation is not None:
        dtype = T.dtype
        idx_R = (..., slice(None, 3), slice(None, 3))
        idx_t = (..., slice(None, 3), 3)
        global_T = common.zeros_as(T, shape=(*batch_shape, 4, 4), xp=xp)
        global_T = common.set(global_T, (..., 3, 3), xp.asarray(1.0, dtype=dtype), xp=xp)

        if global_rotation is not None:
            R_global = SO3.conversions.from_axis_angle_to_rotmat(global_rotation, xp=xp)
            global_T = common.set(global_T, idx_R, R_global, xp=xp)
        else:
            eye3 = common.eye_as(r_g, batch_dims=batch_shape, xp=xp)
            global_T = common.set(global_T, idx_R, eye3, xp=xp)

        if global_translation is not None:
            global_T = common.set(global_T, idx_t, global_translation, xp=xp)

        T = xp.einsum("...ij,...njk->...nik", global_T, T)

    return T


def _forward_skeleton_core(
    xp,
    pose: Float[Array, "B 204"],
    joint_offsets: Float[Array, "J 3"],
    joint_pre_rotations: Float[Array, "J 4"],
    parameter_transform: Float[Array, "D N"],
    kinematic_fronts: list[Front],
    num_joints: int,
    shape_dim: int,
    joint_indices: list[int] | None = None,
) -> tuple[Float[Array, "B J 3"], Float[Array, "B J 3 3"], Float[Array, "B J 1"], Float[Array, "B J 7"]]:
    j_p = _pose_to_joint_params(xp, pose, parameter_transform, num_joints, shape_dim)

    t_l = j_p[..., :3] + joint_offsets
    euler = j_p[..., 3:6]

    q_local = SO3.canonicalize(
        SO3.conversions.from_euler_to_quat(euler, euler_convention="xyz", quat_convention="xyzw", xp=xp),
        convention="xyzw",
        xp=xp,
    )
    q_l = SO3.canonicalize(
        SO3.multiply(joint_pre_rotations, q_local, convention="xyzw", xp=xp), convention="xyzw", xp=xp
    )

    s_l = xp.exp(_LN2 * j_p[..., 6:7])
    t_g, r_g, s_g = _compose_global_trs(xp, t_l, q_l, s_l, kinematic_fronts, num_joints, joint_indices)

    return t_g, r_g, s_g, j_p


def _pose_to_joint_params(
    xp,
    pose: Float[Array, "B 204"],
    parameter_transform: Float[Array, "D N"],
    num_joints: int,
    shape_dim: int,
) -> Float[Array, "B J 7"]:
    """Convert pose vector to per-joint parameters [B, J, 7]."""
    batch_shape = pose.shape[:-1]
    pad = common.zeros_as(pose, shape=(*batch_shape, shape_dim), xp=xp)
    j_p = xp.einsum("dn,...n->...d", parameter_transform, xp.concat([pose, pad], axis=-1))
    return j_p.reshape(*batch_shape, num_joints, 7)


def _compose_global_trs(
    xp,
    t_l: Float[Array, "B J 3"],
    q_l: Float[Array, "B J 4"],
    s_l: Float[Array, "B J 1"],
    kinematic_fronts: list[Front],
    num_joints: int,
    joint_indices: list[int] | None = None,
) -> tuple[Float[Array, "B J 3"], Float[Array, "B J 3 3"], Float[Array, "B J 1"]]:
    r_l = SO3.conversions.from_quat_to_rotmat(q_l, convention="xyzw", xp=xp)

    t_results: list[Float[Array, "B 3"] | None] = [None] * num_joints
    s_results: list[Float[Array, "B 1"] | None] = [None] * num_joints
    r_results: list[Float[Array, "B 3 3"] | None] = [None] * num_joints

    for joints, parents in kinematic_fronts:
        if parents[0] < 0:
            for j in joints:
                t_results[j] = t_l[..., j, :]
                s_results[j] = s_l[..., j, :]
                r_results[j] = r_l[..., j, :, :]
        else:
            for j, p in zip(joints, parents):
                r_results[j] = r_results[p] @ r_l[..., j, :, :]
                s_results[j] = s_results[p] * s_l[..., j, :]
                r_ps = r_results[p] * s_results[p][..., :, None]
                t_ps = xp.squeeze(r_ps @ t_l[..., j, :, None], axis=-1)
                t_results[j] = t_ps + t_results[p]

    if joint_indices is None:
        t_g = xp.stack(t_results, axis=-2)
        s_g = xp.stack(s_results, axis=-2)
        r_g = xp.stack(r_results, axis=-3)
    else:
        t_g = xp.stack([t_results[j] for j in joint_indices], axis=-2)
        s_g = xp.stack([s_results[j] for j in joint_indices], axis=-2)
        r_g = xp.stack([r_results[j] for j in joint_indices], axis=-3)

    return t_g, r_g, s_g


def _trs_to_transforms(
    xp,
    t: Float[Array, "B J 3"],
    r: Float[Array, "B J 3 3"],
    s: Float[Array, "B J 1"],
) -> Float[Array, "B J 4 4"]:
    R = r * s[..., None]
    batch_shape = t.shape[:-2]
    J = t.shape[-2]
    dtype = t.dtype

    T = common.zeros_as(t, shape=(*batch_shape, J, 4, 4), xp=xp)
    idx_R = (..., slice(None, 3), slice(None, 3))
    idx_t = (..., slice(None, 3), 3)
    T = common.set(T, idx_R, R, xp=xp)
    T = common.set(T, idx_t, t, xp=xp)
    T = common.set(T, (..., 3, 3), xp.asarray(1.0, dtype=dtype), xp=xp)
    return T


def _gather_joint_matrices(arr: Array, indices: Int[Array, "V K"]) -> Array:
    V, K = indices.shape
    flat_indices = indices.reshape(-1)
    gathered = arr[..., flat_indices, :, :]
    new_shape = (*arr.shape[:-3], V, K, *arr.shape[-2:])
    return gathered.reshape(new_shape)


def _gather_joint_vectors(arr: Array, indices: Int[Array, "V K"]) -> Array:
    V, K = indices.shape
    flat_indices = indices.reshape(-1)
    gathered = arr[..., flat_indices, :]
    new_shape = (*arr.shape[:-2], V, K, arr.shape[-1])
    return gathered.reshape(new_shape)
