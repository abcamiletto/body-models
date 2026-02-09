"""Backend-agnostic MHR computation using array_api_compat.

Note: Pose correctives are NOT included here - they are PyTorch-only.
The torch.py backend adds pose correctives on top of this core computation.
"""

import math
from typing import Any, Callable

from array_api_compat import get_namespace
from jaxtyping import Float, Int
from nanomanifold import SO3

from .. import common

Array = Any  # Generic array type (numpy, torch, jax)

_LN2 = math.log(2)


def forward_vertices(
    # Model data
    base_vertices: Float[Array, "V 3"],
    blendshape_dirs: Float[Array, "117 V 3"],
    skin_weights: Float[Array, "V K"],
    skin_indices: Int[Array, "V K"],
    joint_offsets: Float[Array, "J 3"],
    joint_pre_rotations: Float[Array, "J 4"],
    parameter_transform: Float[Array, "D N"],
    bind_inv_linear: Float[Array, "J 3 3"],
    bind_inv_translation: Float[Array, "J 3"],
    kinematic_fronts: list[tuple[list[int], list[int]]],
    num_joints: int,
    shape_dim: int,
    expr_dim: int,
    # Inputs
    shape: Float[Array, "B 45"],
    pose: Float[Array, "B 204"],
    expression: Float[Array, "B 72"] | None = None,
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    # Optional pose correctives callback (PyTorch only)
    pose_correctives_fn: Callable[[Float[Array, "B J 7"]], Float[Array, "B V 3"]] | None = None,
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3] in meters."""
    assert shape.ndim == 2 and shape.shape[1] == shape_dim
    assert pose.ndim == 2 and pose.shape[1] == 204
    assert expression is None or (expression.ndim == 2 and expression.shape[1] == expr_dim)
    assert global_rotation is None or (global_rotation.ndim == 2 and global_rotation.shape[1] == 3)
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(shape)
    B = pose.shape[0]
    dtype = shape.dtype

    # Handle expression
    if expression is None:
        expression = xp.zeros((B, expr_dim), dtype=dtype)

    # Broadcast shape/expression if needed
    if shape.shape[0] == 1 and B > 1:
        shape = xp.broadcast_to(shape, (B, shape.shape[1]))
    if expression.shape[0] == 1 and B > 1:
        expression = xp.broadcast_to(expression, (B, expression.shape[1]))

    coeffs = xp.concat([shape, expression], axis=1)

    # Forward skeleton
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

    # Blendshapes
    v_t = base_vertices + xp.einsum("bi,ivk->bvk", coeffs, blendshape_dirs)

    # Apply pose correctives if provided (PyTorch only)
    if pose_correctives_fn is not None:
        v_t = v_t + pose_correctives_fn(j_p)

    # Linear blend skinning
    lin_g = r_g * s_g[..., None]  # [B, J, 3, 3]
    lin = xp.einsum("bjik,jkl->bjil", lin_g, bind_inv_linear)  # [B, J, 3, 3]
    t = xp.einsum("bjik,jk->bji", lin_g, bind_inv_translation) + t_g  # [B, J, 3]

    # Gather skinning matrices per vertex
    lin = _gather_axis1(xp, lin, skin_indices)  # [B, V, K, 3, 3]
    t = _gather_axis1(xp, t, skin_indices)  # [B, V, K, 3]

    # Apply LBS: v' = sum_k w_k * (R_k @ v + t_k)
    v_transformed = xp.einsum("bvkij,bvj->bvki", lin, v_t) + t  # [B, V, K, 3]
    verts = xp.sum(v_transformed * skin_weights[None, :, :, None], axis=2)  # [B, V, 3]

    # Convert to meters
    verts = verts * 0.01

    # Apply global transform
    if global_rotation is not None:
        R = SO3.to_matrix(SO3.from_axis_angle(global_rotation, xp=xp), xp=xp)
        verts = xp.einsum("bij,bvj->bvi", R, verts)
    if global_translation is not None:
        verts = verts + global_translation[:, None]

    return verts


def forward_skeleton(
    # Model data
    joint_offsets: Float[Array, "J 3"],
    joint_pre_rotations: Float[Array, "J 4"],
    parameter_transform: Float[Array, "D N"],
    kinematic_fronts: list[tuple[list[int], list[int]]],
    num_joints: int,
    shape_dim: int,
    # Inputs
    pose: Float[Array, "B 204"],
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
    """Compute skeleton transforms [B, J, 4, 4] in meters."""
    assert pose.ndim == 2 and pose.shape[1] == 204
    assert global_rotation is None or (global_rotation.ndim == 2 and global_rotation.shape[1] == 3)
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(pose)

    t_g, r_g, s_g, _ = _forward_skeleton_core(
        xp=xp,
        pose=pose,
        joint_offsets=joint_offsets,
        joint_pre_rotations=joint_pre_rotations,
        parameter_transform=parameter_transform,
        kinematic_fronts=kinematic_fronts,
        num_joints=num_joints,
        shape_dim=shape_dim,
    )

    # Build 4x4 transforms (in meters)
    T = _trs_to_transforms(xp, t_g * 0.01, r_g, s_g)

    # Apply global transform
    if global_rotation is not None or global_translation is not None:
        B = T.shape[0]
        dtype = T.dtype
        idx_R = (slice(None), slice(None, 3), slice(None, 3))
        idx_t = (slice(None), slice(None, 3), 3)
        global_T = xp.zeros((B, 4, 4), dtype=dtype)
        global_T = common.set(global_T, (slice(None), 3, 3), xp.asarray(1.0, dtype=dtype), xp=xp)

        if global_rotation is not None:
            R_global = SO3.to_matrix(SO3.from_axis_angle(global_rotation, xp=xp), xp=xp)
            global_T = common.set(global_T, idx_R, R_global, xp=xp)
        else:
            eye3 = xp.eye(3, dtype=dtype)
            global_T = common.set(global_T, idx_R, eye3, xp=xp)

        if global_translation is not None:
            global_T = common.set(global_T, idx_t, global_translation, xp=xp)

        # global_T: [B, 4, 4], T: [B, J, 4, 4] -> broadcast global_T @ T for each joint
        T = xp.einsum("bij,bnjk->bnik", global_T, T)

    return T


def _forward_skeleton_core(
    xp,
    pose: Float[Array, "B 204"],
    joint_offsets: Float[Array, "J 3"],
    joint_pre_rotations: Float[Array, "J 4"],
    parameter_transform: Float[Array, "D N"],
    kinematic_fronts: list[tuple[list[int], list[int]]],
    num_joints: int,
    shape_dim: int,
) -> tuple[Float[Array, "B J 3"], Float[Array, "B J 3 3"], Float[Array, "B J 1"], Float[Array, "B J 7"]]:
    """Compute global skeleton transforms from pose.

    Returns (t_g, r_g, s_g, j_p) - translations, rotation matrices, scales, joint params.
    """
    # Convert pose to joint parameters
    j_p = _pose_to_joint_params(xp, pose, parameter_transform, num_joints, shape_dim)

    # Local transforms
    t_l = j_p[..., :3] + joint_offsets  # [B, J, 3]
    euler = j_p[..., 3:6]  # [B, J, 3]

    # Convert euler to quaternion and apply pre-rotation
    q_local = SO3.to_quat_xyzw(SO3.canonicalize(SO3.from_euler(euler, convention="xyz", xp=xp), xp=xp), xp=xp)
    q_l = SO3.canonicalize(SO3.multiply(joint_pre_rotations, q_local, xyzw=True, xp=xp), xyzw=True, xp=xp)

    # Scale from joint params
    s_l = xp.exp(_LN2 * j_p[..., 6:7])  # [B, J, 1]

    # Forward kinematics
    t_g, r_g, s_g = _compose_global_trs(xp, t_l, q_l, s_l, kinematic_fronts, num_joints)

    return t_g, r_g, s_g, j_p


def _pose_to_joint_params(
    xp,
    pose: Float[Array, "B 204"],
    parameter_transform: Float[Array, "D N"],
    num_joints: int,
    shape_dim: int,
) -> Float[Array, "B J 7"]:
    """Convert pose vector to per-joint parameters [B, J, 7]."""
    B = pose.shape[0]
    dtype = pose.dtype
    pad = xp.zeros((B, shape_dim), dtype=dtype)
    j_p = xp.einsum("dn,bn->bd", parameter_transform, xp.concat([pose, pad], axis=-1))
    return j_p.reshape(B, num_joints, 7)


def _compose_global_trs(
    xp,
    t_l: Float[Array, "B J 3"],
    q_l: Float[Array, "B J 4"],
    s_l: Float[Array, "B J 1"],
    kinematic_fronts: list[tuple[list[int], list[int]]],
    num_joints: int,
) -> tuple[Float[Array, "B J 3"], Float[Array, "B J 3 3"], Float[Array, "B J 1"]]:
    """Compose local TRS transforms into global via batched FK."""
    r_l = SO3.to_matrix(q_l, xyzw=True, xp=xp)  # [B, J, 3, 3]

    t_results: list[Float[Array, "B 3"] | None] = [None] * num_joints
    s_results: list[Float[Array, "B 1"] | None] = [None] * num_joints
    r_results: list[Float[Array, "B 3 3"] | None] = [None] * num_joints

    for joints, parents in kinematic_fronts:
        if parents[0] < 0:  # Root joints
            for j in joints:
                t_results[j] = t_l[:, j]
                s_results[j] = s_l[:, j]
                r_results[j] = r_l[:, j]
        else:
            for j, p in zip(joints, parents):
                r_results[j] = r_results[p] @ r_l[:, j]
                s_results[j] = s_results[p] * s_l[:, j]
                r_ps = r_results[p] * s_results[p][:, :, None]
                t_results[j] = xp.squeeze(r_ps @ t_l[:, j, :, None], axis=-1) + t_results[p]

    t_g = xp.stack(t_results, axis=1)
    s_g = xp.stack(s_results, axis=1)
    r_g = xp.stack(r_results, axis=1)

    return t_g, r_g, s_g


def _trs_to_transforms(
    xp,
    t: Float[Array, "B J 3"],
    r: Float[Array, "B J 3 3"],
    s: Float[Array, "B J 1"],
) -> Float[Array, "B J 4 4"]:
    """Convert translation, rotation matrix, scale to 4x4 transforms."""
    R = r * s[..., None]
    B, J = t.shape[:2]
    dtype = t.dtype

    T = xp.zeros((B, J, 4, 4), dtype=dtype)
    idx_R = (..., slice(None, 3), slice(None, 3))
    idx_t = (..., slice(None, 3), 3)
    T = common.set(T, idx_R, R, xp=xp)
    T = common.set(T, idx_t, t, xp=xp)
    T = common.set(T, (..., 3, 3), xp.asarray(1.0, dtype=dtype), xp=xp)
    return T


def _gather_axis1(xp, arr: Array, indices: Int[Array, "V K"]) -> Array:
    """Gather from arr along axis 1 using indices.

    arr: [B, J, ...] -> out: [B, V, K, ...]
    indices: [V, K] with values in range [0, J)
    """
    # Use advanced indexing
    # For shape [B, J, 3, 3] and indices [V, K], we want [B, V, K, 3, 3]
    B = arr.shape[0]
    V, K = indices.shape

    # Flatten indices and gather
    flat_indices = indices.reshape(-1)  # [V*K]

    # Gather: arr[:, flat_indices] -> [B, V*K, ...]
    gathered = arr[:, flat_indices]

    # Reshape to [B, V, K, ...]
    new_shape = (B, V, K) + arr.shape[2:]
    return gathered.reshape(new_shape)


def extract_skeleton_state(
    transforms: Float[Array, "B J 4 4"],
) -> Float[Array, "B J 8"]:
    """Extract skeleton state [t, q, s] from 4x4 transforms.

    Args:
        transforms: World-space 4x4 transform matrices [B, J, 4, 4].

    Returns:
        Skeleton state [B, J, 8] with [translation(3), quaternion_xyzw(4), scale(1)].
    """
    xp = get_namespace(transforms)
    t = transforms[..., :3, 3]
    R = transforms[..., :3, :3]

    # Extract scale (uniform, from first column norm)
    s = xp.linalg.vector_norm(R[..., :, 0], axis=-1, keepdims=True)

    # Extract pure rotation and convert to quaternion
    R_pure = R / s[..., None]
    q = SO3.to_quat_xyzw(SO3.from_matrix(R_pure, xp=xp), xp=xp)

    return xp.concat([t, q, s], axis=-1)


def from_native_args(
    shape: Float[Array, "B 45"],
    expression: Float[Array, "B 72"],
    pose: Float[Array, "B 204"],
) -> dict[str, Array]:
    """Convert native MHR args (shape, expression, pose) to forward_* kwargs."""
    return {"shape": shape, "pose": pose, "expression": expression}


def to_native_outputs(
    vertices: Float[Array, "B V 3"],
    transforms: Float[Array, "B J 4 4"],
) -> dict[str, Array]:
    """Convert forward_* outputs to native MHR format (cm units, skeleton state).

    Args:
        vertices: Mesh vertices [B, V, 3] in meters.
        transforms: Skeleton transforms [B, J, 4, 4] in meters.

    Returns:
        Dict with "vertices" and "joints" in cm units.
    """
    xp = get_namespace(vertices)
    skel_state = extract_skeleton_state(transforms)
    # Scale translation to cm
    t_cm = skel_state[..., :3] * 100
    q = skel_state[..., 3:7]
    s = skel_state[..., 7:8]
    skel_state_cm = xp.concat([t_cm, q, s], axis=-1)
    return {
        "vertices": vertices * 100,
        "joints": skel_state_cm,
    }
