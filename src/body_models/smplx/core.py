"""Backend-agnostic SMPL-X computation using array_api_compat."""

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
    shapedirs: Float[Array, "V 3 S"],
    shapedirs_full: Float[Array, "V_full 3 S"],
    exprdirs: Float[Array, "V 3 E"],
    exprdirs_full: Float[Array, "V_full 3 E"],
    posedirs: Float[Array, "P V*3"],
    lbs_weights: Float[Array, "V 55"],
    J_regressor: Float[Array, "55 V_full"],
    parents: Int[Array, "55"],
    kinematic_fronts: list[tuple[list[int], list[int]]],
    hand_mean: Float[Array, "2 45"],
    rest_pose_y_offset: float,
    # Inputs
    shape: Float[Array, "B 10"],
    body_pose: Float[Array, "B 21 3"],
    hand_pose: Float[Array, "B 30 3"],
    head_pose: Float[Array, "B 3 3"],
    expression: Float[Array, "B 10"] | None = None,
    pelvis_rotation: Float[Array, "B 3"] | None = None,
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    ground_plane: bool = True,
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3]."""
    assert shape.ndim == 2 and shape.shape[1] >= 1
    assert body_pose.ndim == 3 and body_pose.shape[1:] == (21, 3)
    assert hand_pose.ndim == 3 and hand_pose.shape[1:] == (30, 3)
    assert head_pose.ndim == 3 and head_pose.shape[1:] == (3, 3)
    assert expression is None or (expression.ndim == 2 and expression.shape[1] >= 1)
    assert pelvis_rotation is None or (pelvis_rotation.ndim == 2 and pelvis_rotation.shape[1] == 3)
    assert global_rotation is None or (global_rotation.ndim == 2 and global_rotation.shape[1] == 3)
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(shape)
    B = body_pose.shape[0]
    dtype = shape.dtype

    if expression is None:
        expression = xp.zeros((B, 10), dtype=dtype)

    v_t, j_t, pose_matrices, T_world = _forward_core(
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
        hand_mean=hand_mean,
        shape=shape,
        expression=expression,
        body_pose=body_pose.reshape(B, -1),
        hand_pose=hand_pose.reshape(B, -1),
        head_pose=head_pose.reshape(B, -1),
        pelvis_rotation=pelvis_rotation,
        skeleton_only=False,
    )
    assert v_t is not None

    # Precomputed offset to place feet on ground plane
    y_offset = rest_pose_y_offset if ground_plane else 0.0

    # Pose blend shapes
    eye3 = xp.eye(3, dtype=dtype)
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

    # Apply ground plane offset (shift Y up by precomputed amount)
    if y_offset != 0.0:
        offset = xp.zeros((1, 1, 3), dtype=v_posed.dtype)
        offset = common.set(offset, (0, 0, 1), xp.asarray(y_offset, dtype=v_posed.dtype))
        v_posed = v_posed + offset

    return v_posed


def forward_skeleton(
    # Model data
    v_template_full: Float[Array, "V_full 3"],
    shapedirs_full: Float[Array, "V_full 3 S"],
    exprdirs_full: Float[Array, "V_full 3 E"],
    J_regressor: Float[Array, "J V_full"],
    parents: Int[Array, "J"],
    kinematic_fronts: list[tuple[list[int], list[int]]],
    hand_mean: Float[Array, "2 45"],
    rest_pose_y_offset: float,
    # Inputs
    shape: Float[Array, "B 10"],
    body_pose: Float[Array, "B 21 3"],
    hand_pose: Float[Array, "B 30 3"],
    head_pose: Float[Array, "B 3 3"],
    expression: Float[Array, "B 10"] | None = None,
    pelvis_rotation: Float[Array, "B 3"] | None = None,
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    ground_plane: bool = True,
    *,
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
    """Compute skeleton joint transforms [B, J, 4, 4]."""
    assert shape.ndim == 2 and shape.shape[1] >= 1
    assert body_pose.ndim == 3 and body_pose.shape[1:] == (21, 3)
    assert hand_pose.ndim == 3 and hand_pose.shape[1:] == (30, 3)
    assert head_pose.ndim == 3 and head_pose.shape[1:] == (3, 3)
    assert expression is None or (expression.ndim == 2 and expression.shape[1] >= 1)
    assert pelvis_rotation is None or (pelvis_rotation.ndim == 2 and pelvis_rotation.shape[1] == 3)
    assert global_rotation is None or (global_rotation.ndim == 2 and global_rotation.shape[1] == 3)
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(shape)
    B = body_pose.shape[0]
    dtype = shape.dtype

    if expression is None:
        expression = xp.zeros((B, 10), dtype=dtype)

    _, _, _, T_world = _forward_core(
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
        hand_mean=hand_mean,
        shape=shape,
        expression=expression,
        body_pose=body_pose.reshape(B, -1),
        hand_pose=hand_pose.reshape(B, -1),
        head_pose=head_pose.reshape(B, -1),
        pelvis_rotation=pelvis_rotation,
        skeleton_only=True,
    )

    # Precomputed offset to place feet on ground plane
    y_offset = rest_pose_y_offset if ground_plane else 0.0

    # Extract R and t from T_world
    R_world = T_world[..., :3, :3]
    t_world = T_world[..., :3, 3]

    # Apply global transform
    if global_rotation is not None or global_translation is not None:
        R_world, t_world = _apply_global_transform_to_rt(xp, R_world, t_world, global_rotation, global_translation)

    # Apply ground plane offset (shift Y up by precomputed amount)
    if y_offset != 0.0:
        offset = xp.zeros((1, 1, 3), dtype=t_world.dtype)
        offset = common.set(offset, (0, 0, 1), xp.asarray(y_offset, dtype=t_world.dtype))
        t_world = t_world + offset

    # Reconstruct T from R and t
    return _build_transform_matrix(xp, R_world, t_world)


def _forward_core(
    xp,
    v_template: Float[Array, "V 3"] | None,
    v_template_full: Float[Array, "V_full 3"],
    shapedirs: Float[Array, "V 3 S"] | None,
    shapedirs_full: Float[Array, "V_full 3 S"],
    exprdirs: Float[Array, "V 3 E"] | None,
    exprdirs_full: Float[Array, "V_full 3 E"],
    J_regressor: Float[Array, "J V_full"],
    parents: Int[Array, "J"],
    kinematic_fronts: list[tuple[list[int], list[int]]],
    hand_mean: Float[Array, "2 45"],
    shape: Float[Array, "B 10"],
    expression: Float[Array, "B 10"],
    body_pose: Float[Array, "B 63"],
    hand_pose: Float[Array, "B 90"],
    head_pose: Float[Array, "B 9"],
    pelvis_rotation: Float[Array, "B 3"] | None,
    skeleton_only: bool,
) -> tuple[
    Float[Array, "B V 3"] | None,
    Float[Array, "B J 3"],
    Float[Array, "B J 3 3"],
    Float[Array, "B J 4 4"],
]:
    """Core forward pass."""
    B = body_pose.shape[0]
    dtype = shape.dtype

    # Broadcast shape if needed
    if shape.shape[0] == 1 and B > 1:
        shape = xp.broadcast_to(shape, (B, shape.shape[1]))

    # Apply hand pose mean
    lh = hand_pose[:, :45]
    rh = hand_pose[:, 45:]
    hand_pose_adj = xp.concat([lh + hand_mean[0], rh + hand_mean[1]], axis=-1)

    # Build full pose with pelvis rotation
    if pelvis_rotation is None:
        pelvis = xp.zeros((B, 3), dtype=dtype)
    else:
        pelvis = pelvis_rotation
    pose = xp.concat([pelvis, body_pose, head_pose, hand_pose_adj], axis=-1).reshape(B, -1, 3)
    pose_matrices = SO3.to_matrix(SO3.from_axis_angle(pose, xp=xp), xp=xp)

    # Joint locations from full-resolution mesh
    shape_dim = shape.shape[-1]
    expr_dim = expression.shape[-1]
    shape_blend = xp.einsum("bi,vdi->bvd", shape, shapedirs_full[:, :, :shape_dim])
    expr_blend = xp.einsum("bi,vdi->bvd", expression, exprdirs_full[:, :, :expr_dim])
    v_t_full = v_template_full + shape_blend + expr_blend
    j_t = xp.einsum("bvd,jv->bjd", v_t_full, J_regressor)

    # Shape blend shapes for mesh output
    if skeleton_only:
        v_t = None
    else:
        assert v_template is not None and shapedirs is not None and exprdirs is not None
        shape_blend_simp = xp.einsum("bi,vdi->bvd", shape, shapedirs[:, :, :shape_dim])
        expr_blend_simp = xp.einsum("bi,vdi->bvd", expression, exprdirs[:, :, :expr_dim])
        v_t = v_template + shape_blend_simp + expr_blend_simp

    # Forward kinematics
    j0 = j_t[:, 0:1]
    j_rest = j_t[:, 1:] - j_t[:, parents[1:]]
    t_local = xp.concat([j0, j_rest], axis=1)

    T_world = _batched_forward_kinematics(xp, pose_matrices, t_local, kinematic_fronts)

    return v_t, j_t, pose_matrices, T_world


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
    shape: Float[Array, "B 10"],
    expression: Float[Array, "B 10"],
    body_pose: Float[Array, "B 63"],
    hand_pose: Float[Array, "B 90"],
    head_pose: Float[Array, "B 9"],
    pelvis_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
) -> dict[str, Array | None]:
    """Convert native SMPLX args to forward_* kwargs."""
    xp = get_namespace(shape)
    return {
        "shape": shape,
        "body_pose": xp.reshape(body_pose, (-1, 21, 3)),
        "hand_pose": xp.reshape(hand_pose, (-1, 30, 3)),
        "head_pose": xp.reshape(head_pose, (-1, 3, 3)),
        "expression": expression,
        "pelvis_rotation": pelvis_rotation,
        "global_translation": global_translation,
    }


def to_native_outputs(
    vertices: Float[Array, "B V 3"],
    transforms: Float[Array, "B J 4 4"],
) -> dict[str, Array]:
    """Convert forward_* outputs to native SMPLX format."""
    return {
        "vertices": vertices,
        "joints": transforms[..., :3, 3],
    }
