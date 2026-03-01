"""Backend-agnostic SMPL-X computation using array_api_compat."""

from typing import Any

from array_api_compat import get_namespace
from jaxtyping import Float, Int
from nanomanifold import SO3

from .. import common

Array = Any  # Generic array type (numpy, torch, jax)


def forward_vertices(
    # Model data
    v_template: Float[Array, "V 3"],
    shapedirs: Float[Array, "V 3 S"],
    exprdirs: Float[Array, "V 3 E"],
    posedirs: Float[Array, "P V*3"],
    lbs_weights: Float[Array, "V 55"],
    j_template: Float[Array, "55 3"],
    j_shapedirs: Float[Array, "55 3 S"],
    j_exprdirs: Float[Array, "55 3 E"],
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
    assert body_pose.ndim >= 3 and body_pose.shape[-2:] == (21, 3)
    assert hand_pose.ndim >= 3 and hand_pose.shape[-2:] == (30, 3)
    assert head_pose.ndim >= 3 and head_pose.shape[-2:] == (3, 3)
    assert shape.ndim >= 1 and shape.shape[-1] >= 1
    assert expression is None or (expression.ndim >= 1 and expression.shape[-1] >= 1)
    assert pelvis_rotation is None or (pelvis_rotation.ndim >= 1 and pelvis_rotation.shape[-1] == 3)
    assert global_rotation is None or (global_rotation.ndim >= 1 and global_rotation.shape[-1] == 3)
    assert global_translation is None or (global_translation.ndim >= 1 and global_translation.shape[-1] == 3)

    if xp is None:
        xp = get_namespace(shape)
    batch_shape = tuple(body_pose.shape[:-2])
    assert tuple(hand_pose.shape[:-2]) == batch_shape
    assert tuple(head_pose.shape[:-2]) == batch_shape
    assert expression is None or tuple(expression.shape[:-1]) == batch_shape
    assert pelvis_rotation is None or tuple(pelvis_rotation.shape[:-1]) == batch_shape
    assert global_rotation is None or tuple(global_rotation.shape[:-1]) == batch_shape
    assert global_translation is None or tuple(global_translation.shape[:-1]) == batch_shape

    shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))

    if expression is None:
        expression = common.zeros_as(shape, shape=(*batch_shape, 10), xp=xp)

    v_t, j_t, pose_matrices, T_world = _forward_core(
        xp=xp,
        v_template=v_template,
        shapedirs=shapedirs,
        exprdirs=exprdirs,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        j_exprdirs=j_exprdirs,
        parents=parents,
        kinematic_fronts=kinematic_fronts,
        hand_mean=hand_mean,
        shape=shape,
        expression=expression,
        body_pose=body_pose.reshape(*batch_shape, -1),
        hand_pose=hand_pose.reshape(*batch_shape, -1),
        head_pose=head_pose.reshape(*batch_shape, -1),
        pelvis_rotation=pelvis_rotation,
        skeleton_only=False,
    )
    assert v_t is not None

    # Precomputed offset to place feet on ground plane
    y_offset = rest_pose_y_offset if ground_plane else 0.0

    # Pose blend shapes
    eye3 = common.eye_as(pose_matrices, batch_dims=(*batch_shape, 1), xp=xp)
    pose_delta = (pose_matrices[..., 1:, :, :] - eye3).reshape(*batch_shape, -1)
    v_shaped = v_t + (pose_delta @ posedirs).reshape(*batch_shape, -1, 3)

    # Linear blend skinning
    R_world = T_world[..., :3, :3]
    t_world = T_world[..., :3, 3]
    W_R = xp.einsum("vj,...jkl->...vkl", lbs_weights, R_world)
    W_t = xp.einsum("vj,...jk->...vk", lbs_weights, t_world - xp.squeeze(R_world @ j_t[..., None], axis=-1))
    v_posed = xp.squeeze(W_R @ v_shaped[..., None], axis=-1) + W_t

    # Apply global transform
    v_posed = _apply_global_transform(xp, v_posed, global_rotation, global_translation)

    # Apply ground plane offset (shift Y up by precomputed amount)
    if y_offset != 0.0:
        offset = common.zeros_as(v_posed, shape=(1, 1, 3), xp=xp)
        offset = common.set(offset, (0, 0, 1), y_offset, xp=xp)
        v_posed = v_posed + offset

    return v_posed


def forward_skeleton(
    # Model data
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    j_exprdirs: Float[Array, "J 3 E"],
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
    assert body_pose.ndim >= 3 and body_pose.shape[-2:] == (21, 3)
    assert hand_pose.ndim >= 3 and hand_pose.shape[-2:] == (30, 3)
    assert head_pose.ndim >= 3 and head_pose.shape[-2:] == (3, 3)
    assert shape.ndim >= 1 and shape.shape[-1] >= 1
    assert expression is None or (expression.ndim >= 1 and expression.shape[-1] >= 1)
    assert pelvis_rotation is None or (pelvis_rotation.ndim >= 1 and pelvis_rotation.shape[-1] == 3)
    assert global_rotation is None or (global_rotation.ndim >= 1 and global_rotation.shape[-1] == 3)
    assert global_translation is None or (global_translation.ndim >= 1 and global_translation.shape[-1] == 3)

    if xp is None:
        xp = get_namespace(shape)
    batch_shape = tuple(body_pose.shape[:-2])
    assert tuple(hand_pose.shape[:-2]) == batch_shape
    assert tuple(head_pose.shape[:-2]) == batch_shape
    assert expression is None or tuple(expression.shape[:-1]) == batch_shape
    assert pelvis_rotation is None or tuple(pelvis_rotation.shape[:-1]) == batch_shape
    assert global_rotation is None or tuple(global_rotation.shape[:-1]) == batch_shape
    assert global_translation is None or tuple(global_translation.shape[:-1]) == batch_shape

    shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))

    if expression is None:
        expression = common.zeros_as(shape, shape=(*batch_shape, 10), xp=xp)

    _, _, _, T_world = _forward_core(
        xp=xp,
        v_template=None,
        shapedirs=None,
        exprdirs=None,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        j_exprdirs=j_exprdirs,
        parents=parents,
        kinematic_fronts=kinematic_fronts,
        hand_mean=hand_mean,
        shape=shape,
        expression=expression,
        body_pose=body_pose.reshape(*batch_shape, -1),
        hand_pose=hand_pose.reshape(*batch_shape, -1),
        head_pose=head_pose.reshape(*batch_shape, -1),
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
        offset = common.zeros_as(t_world, shape=(1, 1, 3), xp=xp)
        offset = common.set(offset, (0, 0, 1), y_offset, xp=xp)
        t_world = t_world + offset

    # Reconstruct T from R and t
    return _build_transform_matrix(xp, R_world, t_world)


def _forward_core(
    xp,
    v_template: Float[Array, "V 3"] | None,
    shapedirs: Float[Array, "V 3 S"] | None,
    exprdirs: Float[Array, "V 3 E"] | None,
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    j_exprdirs: Float[Array, "J 3 E"],
    parents: Int[Array, "J"],
    kinematic_fronts: list[tuple[list[int], list[int]]],
    hand_mean: Float[Array, "2 45"],
    shape: Float[Array, "*batch 10"],
    expression: Float[Array, "*batch 10"],
    body_pose: Float[Array, "*batch 63"],
    hand_pose: Float[Array, "*batch 90"],
    head_pose: Float[Array, "*batch 9"],
    pelvis_rotation: Float[Array, "*batch 3"] | None,
    skeleton_only: bool,
) -> tuple[
    Float[Array, "*batch V 3"] | None,
    Float[Array, "*batch J 3"],
    Float[Array, "*batch J 3 3"],
    Float[Array, "*batch J 4 4"],
]:
    """Core forward pass."""
    batch_shape = body_pose.shape[:-1]
    shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))

    # Apply hand pose mean
    lh = hand_pose[..., :45]
    rh = hand_pose[..., 45:]
    hand_pose_adj = xp.concat([lh + hand_mean[0], rh + hand_mean[1]], axis=-1)

    # Build full pose with pelvis rotation
    if pelvis_rotation is None:
        pelvis = common.zeros_as(shape, shape=(*batch_shape, 3), xp=xp)
    else:
        pelvis = pelvis_rotation
    pose = xp.concat([pelvis, body_pose, head_pose, hand_pose_adj], axis=-1).reshape(*batch_shape, -1, 3)
    pose_matrices = SO3.conversions.from_axis_angle_to_matrix(pose, xp=xp)

    # Joint locations from precomputed regression matrices
    shape_dim = shape.shape[-1]
    expr_dim = expression.shape[-1]
    params_full = xp.concat([shape, expression], axis=-1)
    j_dirs = xp.concat([j_shapedirs[:, :, :shape_dim], j_exprdirs[:, :, :expr_dim]], axis=-1)
    j_t = j_template + xp.einsum("...p,jdp->...jd", params_full, j_dirs)

    # Shape blend shapes for mesh output
    if skeleton_only:
        v_t = None
    else:
        assert v_template is not None and shapedirs is not None and exprdirs is not None
        dirs_simp = xp.concat([shapedirs[:, :, :shape_dim], exprdirs[:, :, :expr_dim]], axis=-1)
        v_t = v_template + xp.einsum("...i,vdi->...vd", params_full, dirs_simp)

    # Forward kinematics
    j0 = j_t[..., 0:1, :]
    j_rest = j_t[..., 1:, :] - j_t[..., parents[1:], :]
    t_local = xp.concat([j0, j_rest], axis=-2)

    T_world = _batched_forward_kinematics(xp, pose_matrices, t_local, kinematic_fronts)

    return v_t, j_t, pose_matrices, T_world


def _batched_forward_kinematics(
    xp,
    R: Float[Array, "*batch J 3 3"],
    t: Float[Array, "*batch J 3"],
    fronts: list[tuple[list[int], list[int]]],
) -> Float[Array, "*batch J 4 4"]:
    """Batched forward kinematics using precomputed kinematic fronts.

    Uses unified 4x4 homogeneous transforms: one bmm per depth level instead
    of two (R_parent @ R_local + R_parent @ t_local).
    """
    J = R.shape[-3]

    # Build all local 4x4 transforms up front
    T_local = _build_transform_matrix(xp, R, t)

    T_world: list[Float[Array, "*batch 4 4"] | None] = [None] * J

    for joints, parents in fronts:
        if parents[0] < 0:  # Root joints
            for joint in joints:
                T_world[joint] = T_local[..., joint, :, :]
            continue

        T_parent = xp.stack([T_world[i] for i in parents], axis=-3)
        T_cur = T_parent @ T_local[..., joints, :, :]
        for idx, joint in enumerate(joints):
            T_world[joint] = T_cur[..., idx, :, :]

    return xp.stack(T_world, axis=-3)


def _build_transform_matrix(
    xp,
    R: Float[Array, "*batch J 3 3"],
    t: Float[Array, "*batch J 3"],
) -> Float[Array, "*batch J 4 4"]:
    """Build 4x4 transform matrix from R [..., J, 3, 3] and t [..., J, 3]."""
    batch_shape = R.shape[:-3]
    J = R.shape[-3]

    upper = xp.concat([R, t[..., None]], axis=-1)
    bottom = common.zeros_as(upper, shape=(*batch_shape, J, 1, 4), xp=xp)
    bottom = common.set(bottom, (..., 0, 3), 1.0, xp=xp)
    return xp.concat([upper, bottom], axis=-2)


def _apply_global_transform(
    xp,
    points: Float[Array, "*batch N 3"],
    rotation: Float[Array, "*batch 3"] | None,
    translation: Float[Array, "*batch 3"] | None,
) -> Float[Array, "*batch N 3"]:
    """Apply global rotation and translation to points [..., N, 3]."""
    if rotation is not None:
        R = SO3.conversions.from_axis_angle_to_matrix(rotation, xp=xp)
        points = (R @ points.mT).mT
    if translation is not None:
        points = points + translation[..., None, :]
    return points


def _apply_global_transform_to_rt(
    xp,
    R: Float[Array, "*batch J 3 3"],
    t: Float[Array, "*batch J 3"],
    rotation: Float[Array, "*batch 3"] | None,
    translation: Float[Array, "*batch 3"] | None,
) -> tuple[Float[Array, "*batch J 3 3"], Float[Array, "*batch J 3"]]:
    """Apply global rotation and translation to R, t components."""
    if rotation is not None:
        R_global = SO3.conversions.from_axis_angle_to_matrix(rotation, xp=xp)
        # Transform t: R_global @ t
        t = (R_global @ t.mT).mT
        # Transform R: R_global @ R (broadcast R_global over J dimension)
        R = R_global[..., None, :, :] @ R
    if translation is not None:
        t = t + translation[..., None, :]
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
