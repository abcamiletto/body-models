"""Backend-agnostic SMPL computation."""

from typing import Any

from jaxtyping import Float

from body_models import common
from body_models.common import get_namespace
from nanomanifold import SO3

from body_models.rotations import RotationType

Array = Any  # Generic array type (numpy, torch, jax)
Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).


def forward_vertices(
    # Model data
    v_template: Float[Array, "V 3"],
    shapedirs: Float[Array, "V D 10"],
    posedirs: Float[Array, "P V*3"],
    lbs_weights: Float[Array, "V 24"],
    j_template: Float[Array, "24 3"],
    j_shapedirs: Float[Array, "24 3 S"],
    parents: list[int],
    kinematic_fronts: list[Front],
    # Inputs
    shape: Float[Array, "B 10"],
    body_pose: Float[Array, "B 23 N"] | Float[Array, "B 23 3 3"],
    pelvis_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3]."""
    assert shape.ndim == 2 and shape.shape[1] >= 1
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(shape)
    v_shaped, j_t, T_world = forward_unskinned_vertices(
        xp=xp,
        v_template=v_template,
        shapedirs=shapedirs,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        parents=parents,
        kinematic_fronts=kinematic_fronts,
        shape=shape,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        posedirs=posedirs,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
    )
    if vertex_indices is not None:
        vertex_indices = xp.asarray(vertex_indices)
        lbs_weights = lbs_weights[vertex_indices]

    R_world = T_world[..., :3, :3]
    t_world = T_world[..., :3, 3]
    W_R = xp.einsum("vj,bjkl->bvkl", lbs_weights, R_world)
    W_t = xp.einsum("vj,bjk->bvk", lbs_weights, t_world - xp.squeeze(R_world @ j_t[..., None], axis=-1))
    v_posed = xp.squeeze(W_R @ v_shaped[..., None], axis=-1) + W_t

    # Apply global transform
    v_posed = apply_global_transform(xp, v_posed, global_rotation, global_translation, rotation_type)

    return v_posed


def forward_unskinned_vertices(
    # Model data
    v_template: Float[Array, "V 3"],
    shapedirs: Float[Array, "V D 10"],
    posedirs: Float[Array, "P V*3"],
    j_template: Float[Array, "24 3"],
    j_shapedirs: Float[Array, "24 3 S"],
    parents: list[int],
    kinematic_fronts: list[Front],
    # Inputs
    shape: Float[Array, "B 10"],
    body_pose: Float[Array, "B 23 N"] | Float[Array, "B 23 3 3"],
    pelvis_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any = None,
) -> tuple[Float[Array, "B V 3"], Float[Array, "B 24 3"], Float[Array, "B 24 4 4"]]:
    if xp is None:
        xp = get_namespace(shape)
    B = body_pose.shape[0]
    if vertex_indices is not None:
        vertex_indices = xp.asarray(vertex_indices)
        v_template = v_template[vertex_indices]
        shapedirs = shapedirs[vertex_indices]
        posedirs = posedirs.reshape(posedirs.shape[0], -1, 3)[:, vertex_indices].reshape(posedirs.shape[0], -1)

    v_t, j_t, pose_matrices, T_world = _forward_core(
        xp=xp,
        v_template=v_template,
        shapedirs=shapedirs,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        parents=parents,
        kinematic_fronts=kinematic_fronts,
        shape=shape,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        skeleton_only=False,
        rotation_type=rotation_type,
    )

    eye3 = common.eye_as(pose_matrices, batch_dims=(B, 1), xp=xp)
    pose_delta = (pose_matrices[:, 1:] - eye3).reshape(B, -1)
    v_shaped = v_t + (pose_delta @ posedirs).reshape(B, -1, 3)
    return v_shaped, j_t, T_world


def forward_skeleton(
    # Model data
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: list[int],
    kinematic_fronts: list[Front],
    # Inputs
    shape: Float[Array, "B 10"],
    body_pose: Float[Array, "B 23 N"] | Float[Array, "B 23 3 3"],
    pelvis_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
    """Compute skeleton joint transforms [B, J, 4, 4]."""
    assert shape.ndim == 2 and shape.shape[1] >= 1
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(shape)
    active_fronts = kinematic_fronts
    if joint_indices is not None:
        joint_indices = [int(joint) for joint in joint_indices]
        if any(joint < 0 or joint >= len(parents) for joint in joint_indices):
            raise IndexError(f"joint_indices must be in [0, {len(parents)})")

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

    _, _, _, T_world = _forward_core(
        xp=xp,
        v_template=None,
        shapedirs=None,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        parents=parents,
        kinematic_fronts=active_fronts,
        shape=shape,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        skeleton_only=True,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
    )

    if global_rotation is None and global_translation is None:
        return T_world

    if global_rotation is None:
        assert global_translation is not None
        t = T_world[..., :3, 3] + global_translation[:, None]
        return common.set(T_world, (..., slice(None, 3), 3), t, xp=xp)

    R_world = T_world[..., :3, :3]
    t_world = T_world[..., :3, 3]

    R_world, t_world = apply_global_transform_to_rt(
        xp,
        R_world,
        t_world,
        global_rotation,
        global_translation,
        rotation_type,
    )

    return build_transform_matrix(xp, R_world, t_world)


def _forward_core(
    xp,
    v_template: Float[Array, "V 3"] | None,
    shapedirs: Float[Array, "V D 10"] | None,
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: list[int],
    kinematic_fronts: list[Front],
    shape: Float[Array, "B 10"],
    body_pose: Float[Array, "B 23 N"] | Float[Array, "B 23 3 3"],
    pelvis_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None,
    skeleton_only: bool,
    rotation_type: RotationType,
    joint_indices: list[int] | None = None,
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
    body_pose_matrices = SO3.convert(body_pose, src=rotation_type, dst="rotmat", xp=xp)
    if pelvis_rotation is None:
        pelvis_matrices = SO3.identity_as(
            body_pose_matrices,
            batch_dims=(B, 1),
            rotation_type="rotmat",
            xp=xp,
        )
    else:
        pelvis_matrices = SO3.convert(
            pelvis_rotation,
            src=rotation_type,
            dst="rotmat",
            xp=xp,
        )[:, None]
    pose_matrices = xp.concat([pelvis_matrices, body_pose_matrices], axis=1)

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

    T_world = _batched_forward_kinematics(xp, pose_matrices, t_local, kinematic_fronts, joint_indices)

    return v_t, j_t, pose_matrices, T_world


def _batched_forward_kinematics(
    xp,
    R: Float[Array, "B J 3 3"],
    t: Float[Array, "B J 3"],
    fronts: list[Front],
    joint_indices: list[int] | None = None,
) -> Float[Array, "B J 4 4"]:
    """Batched forward kinematics using precomputed kinematic fronts.

    Uses unified 4x4 homogeneous transforms: one bmm per depth level instead
    of two (R_parent @ R_local + R_parent @ t_local).
    """
    _, J = R.shape[:2]

    # Build all local 4x4 transforms up front
    T_local = build_transform_matrix(xp, R, t)

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

    if joint_indices is None:
        return xp.stack(T_world, axis=1)
    return xp.stack([T_world[j] for j in joint_indices], axis=1)


def build_transform_matrix(
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


def apply_global_transform(
    xp,
    points: Float[Array, "B N 3"],
    rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None,
    translation: Float[Array, "B 3"] | None,
    rotation_type: RotationType,
) -> Float[Array, "B N 3"]:
    """Apply global rotation and translation to points [B, N, 3]."""
    if rotation is not None:
        R = SO3.convert(rotation, src=rotation_type, dst="rotmat", xp=xp)
        points = (R @ points.mT).mT
    if translation is not None:
        points = points + translation[:, None]
    return points


def apply_global_transform_to_rt(
    xp,
    R: Float[Array, "B J 3 3"],
    t: Float[Array, "B J 3"],
    rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None,
    translation: Float[Array, "B 3"] | None,
    rotation_type: RotationType,
) -> tuple[Float[Array, "B J 3 3"], Float[Array, "B J 3"]]:
    """Apply global rotation and translation to R, t components."""
    if rotation is not None:
        R_global = SO3.convert(rotation, src=rotation_type, dst="rotmat", xp=xp)
        # Transform t: R_global @ t
        t = (R_global @ t.mT).mT
        # Transform R: R_global @ R (broadcast R_global over J dimension)
        R = R_global[:, None] @ R
    if translation is not None:
        t = t + translation[:, None]
    return R, t
