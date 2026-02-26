"""Backend-agnostic SMPL computation using array_api_compat."""

from typing import Any

from array_api_compat import get_namespace
from jaxtyping import Float, Int
from nanomanifold import SO3

from .. import common

Array = Any  # Generic array type (numpy, torch, jax)


def forward_vertices(
    # Model data
    v_template: Float[Array, "V 3"],
    shapedirs: Float[Array, "V D 10"],
    posedirs: Float[Array, "P V*3"],
    lbs_weights: Float[Array, "V 24"],
    j_template: Float[Array, "24 3"],
    j_shapedirs: Float[Array, "24 3 S"],
    parents: Int[Array, "24"],
    kinematic_fronts: list[tuple[list[int], list[int]]],
    rest_pose_y_offset: float,
    # Inputs
    shape: Float[Array, "B 10"],
    body_pose: Float[Array, "B 23 3"],
    pelvis_rotation: Float[Array, "B 3"] | None = None,
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    ground_plane: bool = True,
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3]."""
    assert shape.ndim == 2 and shape.shape[1] >= 1
    assert body_pose.ndim == 3 and body_pose.shape[1:] == (23, 3)
    assert pelvis_rotation is None or (pelvis_rotation.ndim == 2 and pelvis_rotation.shape[1] == 3)
    assert global_rotation is None or (global_rotation.ndim == 2 and global_rotation.shape[1] == 3)
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(shape)
    B = body_pose.shape[0]

    v_t, j_t, pose_matrices, T_world = _forward_core(
        xp=xp,
        v_template=v_template,
        shapedirs=shapedirs,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        parents=parents,
        kinematic_fronts=kinematic_fronts,
        shape=shape,
        body_pose=body_pose.reshape(B, -1),
        pelvis_rotation=pelvis_rotation,
        skeleton_only=False,
    )

    # Precomputed offset to place feet on ground plane
    y_offset = rest_pose_y_offset if ground_plane else 0.0

    # Pose blend shapes
    eye3 = common.eye_as(pose_matrices, batch_dims=(B, 1), xp=xp)
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
        offset = common.zeros_as(v_posed, shape=(1, 1, 3), xp=xp)
        offset = common.set(offset, (0, 0, 1), xp.asarray(y_offset, dtype=v_posed.dtype), xp=xp)
        v_posed = v_posed + offset

    return v_posed


def forward_skeleton(
    # Model data
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: Int[Array, "J"],
    kinematic_fronts: list[tuple[list[int], list[int]]],
    rest_pose_y_offset: float,
    # Inputs
    shape: Float[Array, "B 10"],
    body_pose: Float[Array, "B 23 3"],
    pelvis_rotation: Float[Array, "B 3"] | None = None,
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    ground_plane: bool = True,
    *,
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
    """Compute skeleton joint transforms [B, J, 4, 4]."""
    assert shape.ndim == 2 and shape.shape[1] >= 1
    assert body_pose.ndim == 3 and body_pose.shape[1:] == (23, 3)
    assert pelvis_rotation is None or (pelvis_rotation.ndim == 2 and pelvis_rotation.shape[1] == 3)
    assert global_rotation is None or (global_rotation.ndim == 2 and global_rotation.shape[1] == 3)
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(shape)
    B = body_pose.shape[0]

    _, _, _, T_world = _forward_core(
        xp=xp,
        v_template=None,
        shapedirs=None,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        parents=parents,
        kinematic_fronts=kinematic_fronts,
        shape=shape,
        body_pose=body_pose.reshape(B, -1),
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
        offset = common.set(offset, (0, 0, 1), xp.asarray(y_offset, dtype=t_world.dtype), xp=xp)
        t_world = t_world + offset

    # Reconstruct T from R and t
    return _build_transform_matrix(xp, R_world, t_world)


def _forward_core(
    xp,
    v_template: Float[Array, "V 3"] | None,
    shapedirs: Float[Array, "V D 10"] | None,
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: Int[Array, "J"],
    kinematic_fronts: list[tuple[list[int], list[int]]],
    shape: Float[Array, "B 10"],
    body_pose: Float[Array, "B 69"],
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

    # Broadcast shape if needed
    if shape.shape[0] == 1 and B > 1:
        shape = xp.broadcast_to(shape, (B, shape.shape[1]))

    # Build full pose with pelvis rotation
    if pelvis_rotation is None:
        pelvis = common.zeros_as(shape, shape=(B, 3), xp=xp)
    else:
        pelvis = pelvis_rotation
    pose = xp.concat([pelvis, body_pose], axis=-1).reshape(B, -1, 3)
    pose_matrices = SO3.conversions.from_axis_angle_to_matrix(pose, xp=xp)

    # Joint locations from precomputed regression matrices
    j_t = j_template + xp.einsum("...p,jdp->...jd", shape, j_shapedirs[:, :, : shape.shape[-1]])

    # Shape blend shapes for mesh output
    if skeleton_only:
        v_t = None
    else:
        assert v_template is not None and shapedirs is not None
        v_t = v_template + xp.einsum("bi,vdi->bvd", shape, shapedirs[:, :, : shape.shape[-1]])

    # Forward kinematics
    # Build t_local using concatenation
    j0 = j_t[:, 0:1]  # [B, 1, 3]
    j_rest = j_t[:, 1:] - j_t[:, parents[1:]]  # [B, J-1, 3]
    t_local = xp.concat([j0, j_rest], axis=1)  # [B, J, 3]

    T_world = _batched_forward_kinematics(xp, pose_matrices, t_local, kinematic_fronts)

    return v_t, j_t, pose_matrices, T_world


def _batched_forward_kinematics(
    xp,
    R: Float[Array, "B J 3 3"],
    t: Float[Array, "B J 3"],
    fronts: list[tuple[list[int], list[int]]],
) -> Float[Array, "B J 4 4"]:
    """Batched forward kinematics using precomputed kinematic fronts.

    Uses unified 4x4 homogeneous transforms: one bmm per depth level instead
    of two (R_parent @ R_local + R_parent @ t_local).
    """
    _, J = R.shape[:2]

    # Build all local 4x4 transforms up front
    T_local = _build_transform_matrix(xp, R, t)

    T_world: list[Float[Array, "B 4 4"] | None] = [None] * J

    for joints, parents in fronts:
        if parents[0] < 0:  # Root joints
            for joint in joints:
                T_world[joint] = T_local[:, joint]
            continue

        T_parent = xp.stack([T_world[i] for i in parents], axis=1)
        T_cur = T_parent @ T_local[:, joints]
        for idx, joint in enumerate(joints):
            T_world[joint] = T_cur[:, idx]

    return xp.stack(T_world, axis=1)


def _build_transform_matrix(
    xp,
    R: Float[Array, "B J 3 3"],
    t: Float[Array, "B J 3"],
) -> Float[Array, "B J 4 4"]:
    """Build 4x4 transform matrix from R [B, J, 3, 3] and t [B, J, 3]."""
    B, J = R.shape[:2]
    dtype = R.dtype

    T = common.zeros_as(R, shape=(B, J, 4, 4), xp=xp)
    idx_R = (..., slice(None, 3), slice(None, 3))
    idx_t = (..., slice(None, 3), 3)
    T = common.set(T, idx_R, R, xp=xp)
    T = common.set(T, idx_t, t, xp=xp)
    T = common.set(T, (..., 3, 3), xp.asarray(1.0, dtype=dtype), xp=xp)
    return T


def _apply_global_transform(
    xp,
    points: Float[Array, "B N 3"],
    rotation: Float[Array, "B 3"] | None,
    translation: Float[Array, "B 3"] | None,
) -> Float[Array, "B N 3"]:
    """Apply global rotation and translation to points [B, N, 3]."""
    if rotation is not None:
        R = SO3.conversions.from_axis_angle_to_matrix(rotation, xp=xp)
        points = (R @ points.mT).mT
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
        R_global = SO3.conversions.from_axis_angle_to_matrix(rotation, xp=xp)
        # Transform t: R_global @ t
        t = (R_global @ t.mT).mT
        # Transform R: R_global @ R (broadcast R_global over J dimension)
        R = R_global[:, None] @ R
    if translation is not None:
        t = t + translation[:, None]
    return R, t


def from_native_args(
    shape: Float[Array, "B 10"],
    body_pose: Float[Array, "B 69"],
    pelvis_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
) -> dict[str, Array | None]:
    """Convert native SMPL args to forward_* kwargs."""
    xp = get_namespace(shape)
    return {
        "shape": shape,
        "body_pose": xp.reshape(body_pose, (-1, 23, 3)),
        "pelvis_rotation": pelvis_rotation,
        "global_translation": global_translation,
    }


def to_native_outputs(
    vertices: Float[Array, "B V 3"],
    transforms: Float[Array, "B J 4 4"],
) -> dict[str, Array]:
    """Convert forward_* outputs to native SMPL format."""
    return {
        "vertices": vertices,
        "joints": transforms[..., :3, 3],
    }
