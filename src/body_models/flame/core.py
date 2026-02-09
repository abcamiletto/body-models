"""Backend-agnostic FLAME computation using array_api_compat."""

from typing import Any

import numpy as np
from array_api_compat import get_namespace
from jaxtyping import Float, Int
from nanomanifold import SO3

from .. import common

Array = Any  # Generic array type (numpy, torch, jax)


def forward_vertices(
    # Model data
    v_template: Float[Array, "V 3"],
    v_template_full: Float[Array, "V_full 3"],
    shapedirs: Float[Array, "V 3 N_shape"],
    shapedirs_full: Float[Array, "V_full 3 N_shape"],
    exprdirs: Float[Array, "V 3 N_expr"],
    exprdirs_full: Float[Array, "V_full 3 N_expr"],
    posedirs: Float[Array, "P V*3"],
    lbs_weights: Float[Array, "V 5"],
    J_regressor: Float[Array, "5 V_full"],
    parents: Int[Array, "5"],
    kinematic_fronts: list[tuple[list[int], list[int]]],
    # Inputs
    shape: Float[Array, "B N_shape"],
    expression: Float[Array, "B N_expr"],
    pose: Float[Array, "B 4 3"],
    head_rotation: Float[Array, "B 3"] | None = None,
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    ground_plane: bool = True,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3]."""
    assert shape.ndim == 2 and shape.shape[1] >= 1
    assert expression.ndim == 2 and expression.shape[1] >= 1
    assert pose.ndim == 3 and pose.shape[1:] == (4, 3)
    assert head_rotation is None or (head_rotation.ndim == 2 and head_rotation.shape[1] == 3)
    assert global_rotation is None or (global_rotation.ndim == 2 and global_rotation.shape[1] == 3)
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    xp = get_namespace(shape)
    B = pose.shape[0]

    v_t, j_t, pose_matrices, T_world, ground_offset = _forward_core(
        xp=xp,
        v_template=v_template,
        v_template_full=v_template_full,
        shapedirs=shapedirs,
        shapedirs_full=shapedirs_full,
        exprdirs=exprdirs,
        exprdirs_full=exprdirs_full,
        J_regressor=J_regressor,
        parents=parents,
        kinematic_fronts=kinematic_fronts,
        shape=shape,
        expression=expression,
        pose=pose.reshape(B, -1),
        head_rotation=head_rotation,
        ground_plane=ground_plane,
        skeleton_only=False,
    )
    assert v_t is not None  # guaranteed when skeleton_only=False

    # Pose blend shapes
    eye3 = xp.eye(3, dtype=shape.dtype)
    pose_delta = (pose_matrices[:, 1:] - eye3).reshape(B, -1)
    v_shaped = v_t + (pose_delta @ posedirs).reshape(B, -1, 3)

    # Linear blend skinning
    R_world = T_world[..., :3, :3]
    t_world = T_world[..., :3, 3]
    W_R = xp.einsum("vj,bjkl->bvkl", lbs_weights, R_world)
    W_t = xp.einsum("vj,bjk->bvk", lbs_weights, t_world - xp.squeeze(R_world @ j_t[..., None], axis=-1))
    v_posed = xp.squeeze(W_R @ v_shaped[..., None], axis=-1) + W_t

    # Apply global transform
    v_posed = _apply_global_transform(xp, v_posed, global_rotation, global_translation)

    # Apply ground offset
    return v_posed + ground_offset[:, None]


def forward_skeleton(
    # Model data
    v_template_full: Float[Array, "V_full 3"],
    shapedirs_full: Float[Array, "V_full 3 N_shape"],
    exprdirs_full: Float[Array, "V_full 3 N_expr"],
    J_regressor: Float[Array, "5 V_full"],
    parents: Int[Array, "5"],
    kinematic_fronts: list[tuple[list[int], list[int]]],
    # Inputs
    shape: Float[Array, "B N_shape"],
    expression: Float[Array, "B N_expr"],
    pose: Float[Array, "B 4 3"],
    head_rotation: Float[Array, "B 3"] | None = None,
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    ground_plane: bool = True,
) -> Float[Array, "B 5 4 4"]:
    """Compute skeleton joint transforms [B, 5, 4, 4]."""
    assert shape.ndim == 2 and shape.shape[1] >= 1
    assert expression.ndim == 2 and expression.shape[1] >= 1
    assert pose.ndim == 3 and pose.shape[1:] == (4, 3)
    assert head_rotation is None or (head_rotation.ndim == 2 and head_rotation.shape[1] == 3)
    assert global_rotation is None or (global_rotation.ndim == 2 and global_rotation.shape[1] == 3)
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    xp = get_namespace(shape)
    B = pose.shape[0]

    _, _, _, T_world, ground_offset = _forward_core(
        xp=xp,
        v_template=None,
        v_template_full=v_template_full,
        shapedirs=None,
        shapedirs_full=shapedirs_full,
        exprdirs=None,
        exprdirs_full=exprdirs_full,
        J_regressor=J_regressor,
        parents=parents,
        kinematic_fronts=kinematic_fronts,
        shape=shape,
        expression=expression,
        pose=pose.reshape(B, -1),
        head_rotation=head_rotation,
        ground_plane=ground_plane,
        skeleton_only=True,
    )

    # Extract R and t from T_world
    R_world = T_world[..., :3, :3]
    t_world = T_world[..., :3, 3]

    # Apply global transform
    if global_rotation is not None or global_translation is not None:
        R_world, t_world = _apply_global_transform_to_rt(xp, R_world, t_world, global_rotation, global_translation)

    # Apply ground offset
    t_world = t_world + ground_offset[:, None]

    # Reconstruct T from R and t
    return _build_transform_matrix(xp, R_world, t_world)


def _forward_core(
    xp,
    v_template: Float[Array, "V 3"] | None,
    v_template_full: Float[Array, "V_full 3"],
    shapedirs: Float[Array, "V 3 N_shape"] | None,
    shapedirs_full: Float[Array, "V_full 3 N_shape"],
    exprdirs: Float[Array, "V 3 N_expr"] | None,
    exprdirs_full: Float[Array, "V_full 3 N_expr"],
    J_regressor: Float[Array, "5 V_full"],
    parents: Int[Array, "5"],
    kinematic_fronts: list[tuple[list[int], list[int]]],
    shape: Float[Array, "B N_shape"],
    expression: Float[Array, "B N_expr"],
    pose: Float[Array, "B 12"],  # 4 joints * 3
    head_rotation: Float[Array, "B 3"] | None,
    ground_plane: bool,
    skeleton_only: bool,
) -> tuple[
    Float[Array, "B V 3"] | None,
    Float[Array, "B 5 3"],
    Float[Array, "B 5 3 3"],
    Float[Array, "B 5 4 4"],
    Float[Array, "B 3"],
]:
    """Core forward pass."""
    B = pose.shape[0]
    dtype = shape.dtype

    # Broadcast shape if needed
    if shape.shape[0] == 1 and B > 1:
        shape = xp.broadcast_to(shape, (B, shape.shape[1]))

    # Build full pose (root joint uses head_rotation if provided)
    root = head_rotation if head_rotation is not None else xp.zeros((B, 3), dtype=dtype)
    full_pose = xp.concat([root, pose], axis=-1).reshape(B, -1, 3)
    pose_matrices = SO3.to_matrix(SO3.from_axis_angle(full_pose, xp=xp), xp=xp)

    # Joint locations from full-resolution mesh
    shape_dim = min(shape.shape[-1], shapedirs_full.shape[-1])
    expr_dim = min(expression.shape[-1], exprdirs_full.shape[-1])
    v_t_full = v_template_full + xp.einsum("bi,vdi->bvd", shape[:, :shape_dim], shapedirs_full[:, :, :shape_dim])
    v_t_full = v_t_full + xp.einsum("bi,vdi->bvd", expression[:, :expr_dim], exprdirs_full[:, :, :expr_dim])
    j_t = xp.einsum("bvd,jv->bjd", v_t_full, J_regressor)

    # Compute ground offset
    if ground_plane:
        min_y = xp.min(v_t_full[..., 1], axis=-1)
        zeros = xp.zeros((B,), dtype=dtype)
        ground_offset = xp.stack([zeros, -min_y, zeros], axis=-1)
    else:
        ground_offset = xp.zeros((B, 3), dtype=dtype)

    # Shape blend shapes for mesh output
    if skeleton_only:
        v_t = None
    else:
        assert v_template is not None and shapedirs is not None and exprdirs is not None
        v_t = v_template + xp.einsum("bi,vdi->bvd", shape[:, :shape_dim], shapedirs[:, :, :shape_dim])
        v_t = v_t + xp.einsum("bi,vdi->bvd", expression[:, :expr_dim], exprdirs[:, :, :expr_dim])

    # Forward kinematics
    # Build t_local using concatenation
    j0 = j_t[:, 0:1]  # [B, 1, 3]
    j_rest = j_t[:, 1:] - j_t[:, parents[1:]]  # [B, J-1, 3]
    t_local = xp.concat([j0, j_rest], axis=1)  # [B, J, 3]

    T_world = _batched_forward_kinematics(xp, pose_matrices, t_local, kinematic_fronts)

    return v_t, j_t, pose_matrices, T_world, ground_offset


def _batched_forward_kinematics(
    xp,
    R: Float[Array, "B J 3 3"],
    t: Float[Array, "B J 3"],
    fronts: list[tuple[list[int], list[int]]],
) -> Float[Array, "B J 4 4"]:
    """Batched forward kinematics using precomputed kinematic fronts."""
    _, J = R.shape[:2]

    R_world: list[Float[Array, "B 3 3"] | None] = [None] * J
    t_world: list[Float[Array, "B 3"] | None] = [None] * J

    for joints, parents in fronts:
        if parents[0] < 0:  # Root joints
            for joint in joints:
                R_world[joint] = R[:, joint]
                t_world[joint] = t[:, joint]
            continue

        R_parent = xp.stack([R_world[i] for i in parents], axis=1)
        t_parent = xp.stack([t_world[i] for i in parents], axis=1)
        R_local = R[:, joints]
        t_local = t[:, joints]

        R_cur = R_parent @ R_local
        t_cur = t_parent + xp.squeeze(R_parent @ t_local[..., None], axis=-1)
        for idx, joint in enumerate(joints):
            R_world[joint] = R_cur[:, idx]
            t_world[joint] = t_cur[:, idx]

    R_world_stacked = xp.stack(R_world, axis=1)
    t_world_stacked = xp.stack(t_world, axis=1)

    return _build_transform_matrix(xp, R_world_stacked, t_world_stacked)


def _build_transform_matrix(
    xp,
    R: Float[Array, "B J 3 3"],
    t: Float[Array, "B J 3"],
) -> Float[Array, "B J 4 4"]:
    """Build 4x4 transform matrix from R [B, J, 3, 3] and t [B, J, 3]."""
    B, J = R.shape[:2]
    dtype = R.dtype

    T = xp.zeros((B, J, 4, 4), dtype=dtype)
    T = common.set(T, np.index_exp[..., :3, :3], R)
    T = common.set(T, np.index_exp[..., :3, 3], t)
    T = common.set(T, np.index_exp[..., 3, 3], xp.asarray(1.0, dtype=dtype))
    return T


def _apply_global_transform(
    xp,
    points: Float[Array, "B N 3"],
    rotation: Float[Array, "B 3"] | None,
    translation: Float[Array, "B 3"] | None,
) -> Float[Array, "B N 3"]:
    """Apply global rotation and translation to points [B, N, 3]."""
    if rotation is not None:
        R = SO3.to_matrix(SO3.from_axis_angle(rotation, xp=xp), xp=xp)
        points = xp.permute_dims(R @ xp.permute_dims(points, (0, 2, 1)), (0, 2, 1))
    if translation is not None:
        points = points + translation[:, None]
    return points


def _apply_global_transform_to_rt(
    xp,
    R: Float[Array, "B J 3 3"],
    t: Float[Array, "B J 3"],
    rotation: Float[Array, "B 3"] | None,
    translation: Float[Array, "B 3"] | None,
) -> tuple[Float[Array, "B J 3 3"], Float[Array, "B J 3"]]:
    """Apply global rotation and translation to R, t components."""
    if rotation is not None:
        R_global = SO3.to_matrix(SO3.from_axis_angle(rotation, xp=xp), xp=xp)
        # Transform t: R_global @ t
        t = xp.permute_dims(R_global @ xp.permute_dims(t, (0, 2, 1)), (0, 2, 1))
        # Transform R: R_global @ R (broadcast R_global over J dimension)
        R = R_global[:, None] @ R
    if translation is not None:
        t = t + translation[:, None]
    return R, t


def from_native_args(
    shape: Float[Array, "B N_shape"],
    expression: Float[Array, "B N_expr"],
    pose: Float[Array, "B 12"],
    head_rotation: Float[Array, "B 3"] | None = None,
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
) -> dict[str, Array | None]:
    """Convert native FLAME args to forward_* kwargs.

    Native format uses flat pose tensor.
    API format uses reshaped pose [B, 4, 3].
    """
    xp = get_namespace(shape)
    return {
        "shape": shape,
        "expression": expression,
        "pose": xp.reshape(pose, (-1, 4, 3)),
        "head_rotation": head_rotation,
        "global_rotation": global_rotation,
        "global_translation": global_translation,
    }


def to_native_outputs(
    vertices: Float[Array, "B V 3"],
    transforms: Float[Array, "B J 4 4"],
) -> dict[str, Array]:
    """Convert forward_* outputs to native FLAME format.

    Native format returns joint positions instead of transforms.
    Use ground_plane=False in the FLAME constructor if you need outputs
    compatible with the official smplx library.

    Args:
        vertices: [B, V, 3] mesh vertices.
        transforms: [B, J, 4, 4] joint transforms.
    """
    return {
        "vertices": vertices,
        "joints": transforms[..., :3, 3],
    }
