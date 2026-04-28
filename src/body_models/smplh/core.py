"""Backend-agnostic SMPL-H computation."""

from typing import Any

from jaxtyping import Float

from .. import common
from ..common import get_namespace
from nanomanifold import SO3

from ..rotations import RotationType, is_rotmat_type

Array = Any  # Generic array type (numpy, torch, jax)
Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).


def forward_vertices(
    # Model data
    v_template: Float[Array, "V 3"],
    shapedirs: Float[Array, "V 3 S"],
    posedirs: Float[Array, "P V*3"],
    lbs_weights: Float[Array, "V 52"],
    j_template: Float[Array, "52 3"],
    j_shapedirs: Float[Array, "52 3 S"],
    parents: list[int],
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "2 45"],
    # Inputs
    shape: Float[Array, "B 10"],
    body_pose: Float[Array, "B 21 N"] | Float[Array, "B 21 3 3"],
    hand_pose: Float[Array, "B 30 N"] | Float[Array, "B 30 3 3"],
    pelvis_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3]."""
    assert shape.ndim >= 1 and shape.shape[-1] >= 1
    assert global_translation is None or (global_translation.ndim >= 1 and global_translation.shape[-1] == 3)

    if xp is None:
        xp = get_namespace(shape)
    if vertex_indices is not None:
        vertex_indices = xp.asarray(vertex_indices)
        v_template = v_template[vertex_indices]
        shapedirs = shapedirs[vertex_indices]
        lbs_weights = lbs_weights[vertex_indices]
        posedirs = posedirs.reshape(posedirs.shape[0], -1, 3)[:, vertex_indices].reshape(posedirs.shape[0], -1)
    pose_ndim = 3 if is_rotmat_type(rotation_type) else 2
    batch_shape = tuple(body_pose.shape[:-pose_ndim])
    assert tuple(hand_pose.shape[:-pose_ndim]) == batch_shape

    shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))

    v_t, j_t, pose_matrices, T_world = _forward_core(
        xp=xp,
        v_template=v_template,
        shapedirs=shapedirs,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        parents=parents,
        kinematic_fronts=kinematic_fronts,
        hand_mean=hand_mean,
        shape=shape,
        body_pose=body_pose,
        hand_pose=hand_pose,
        pelvis_rotation=pelvis_rotation,
        skeleton_only=False,
        rotation_type=rotation_type,
    )
    assert v_t is not None

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
    v_posed = _apply_global_transform(xp, v_posed, global_rotation, global_translation, rotation_type)

    return v_posed


def forward_skeleton(
    # Model data
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: list[int],
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "2 45"],
    # Inputs
    shape: Float[Array, "B 10"],
    body_pose: Float[Array, "B 21 N"] | Float[Array, "B 21 3 3"],
    hand_pose: Float[Array, "B 30 N"] | Float[Array, "B 30 3 3"],
    pelvis_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
    """Compute skeleton joint transforms [B, J, 4, 4]."""
    assert shape.ndim >= 1 and shape.shape[-1] >= 1
    assert global_translation is None or (global_translation.ndim >= 1 and global_translation.shape[-1] == 3)

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
    pose_ndim = 3 if is_rotmat_type(rotation_type) else 2
    batch_shape = tuple(body_pose.shape[:-pose_ndim])
    assert tuple(hand_pose.shape[:-pose_ndim]) == batch_shape

    shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))

    _, _, _, T_world = _forward_core(
        xp=xp,
        v_template=None,
        shapedirs=None,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        parents=parents,
        kinematic_fronts=active_fronts,
        hand_mean=hand_mean,
        shape=shape,
        body_pose=body_pose,
        hand_pose=hand_pose,
        pelvis_rotation=pelvis_rotation,
        skeleton_only=True,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
    )

    # Extract R and t from T_world
    R_world = T_world[..., :3, :3]
    t_world = T_world[..., :3, 3]

    # Apply global transform
    if global_rotation is not None or global_translation is not None:
        R_world, t_world = _apply_global_transform_to_rt(
            xp,
            R_world,
            t_world,
            global_rotation,
            global_translation,
            rotation_type,
        )

    # Reconstruct T from R and t
    return _build_transform_matrix(xp, R_world, t_world)


def _forward_core(
    xp,
    v_template: Float[Array, "V 3"] | None,
    shapedirs: Float[Array, "V 3 S"] | None,
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: list[int],
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "2 45"],
    shape: Float[Array, "*batch 10"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    skeleton_only: bool,
    rotation_type: RotationType,
    joint_indices: list[int] | None = None,
) -> tuple[
    Float[Array, "*batch V 3"] | None,
    Float[Array, "*batch J 3"],
    Float[Array, "*batch J 3 3"],
    Float[Array, "*batch J 4 4"],
]:
    """Core forward pass."""
    pose_ndim = 3 if is_rotmat_type(rotation_type) else 2
    batch_shape = body_pose.shape[:-pose_ndim]
    shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))

    # Apply hand pose mean
    if rotation_type == "axis_angle":
        hand_pose_axis_angle = hand_pose
    else:
        # SMPL-H hand pose mean is defined in axis-angle coordinates.
        hand_pose_axis_angle = SO3.convert(hand_pose, src=rotation_type, dst="axis_angle", xp=xp)
    lh = hand_pose_axis_angle[..., :15, :] + hand_mean[0].reshape(15, 3)
    rh = hand_pose_axis_angle[..., 15:, :] + hand_mean[1].reshape(15, 3)
    hand_pose_adj = xp.concat([lh, rh], axis=-2)

    # Build full pose with pelvis rotation
    if pelvis_rotation is None:
        pelvis_matrices = SO3.identity_as(
            body_pose,
            batch_dims=(*batch_shape, 1),
            rotation_type="rotmat",
            xp=xp,
        )
    else:
        pelvis_matrices = SO3.convert(pelvis_rotation, src=rotation_type, dst="rotmat", xp=xp)[..., None, :, :]
    body_matrices = SO3.convert(body_pose, src=rotation_type, dst="rotmat", xp=xp)
    hand_matrices = SO3.convert(hand_pose_adj, src="axis_angle", dst="rotmat", xp=xp)
    pose_matrices = xp.concat([pelvis_matrices, body_matrices, hand_matrices], axis=-3)

    # Joint locations from precomputed regression matrices
    shape_dim = shape.shape[-1]
    j_t = j_template + xp.einsum("...p,jdp->...jd", shape, j_shapedirs[:, :, :shape_dim])

    # Shape blend shapes for mesh output
    if skeleton_only:
        v_t = None
    else:
        assert v_template is not None and shapedirs is not None
        v_t = v_template + xp.einsum("...i,vdi->...vd", shape, shapedirs[:, :, :shape_dim])

    # Forward kinematics
    j0 = j_t[..., 0:1, :]
    j_rest = j_t[..., 1:, :] - j_t[..., parents[1:], :]
    t_local = xp.concat([j0, j_rest], axis=-2)

    T_world = _batched_forward_kinematics(xp, pose_matrices, t_local, kinematic_fronts, joint_indices)

    return v_t, j_t, pose_matrices, T_world


def _batched_forward_kinematics(
    xp,
    R: Float[Array, "*batch J 3 3"],
    t: Float[Array, "*batch J 3"],
    fronts: list[Front],
    joint_indices: list[int] | None = None,
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

    if joint_indices is None:
        return xp.stack(T_world, axis=-3)
    return xp.stack([T_world[j] for j in joint_indices], axis=-3)


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
    rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    translation: Float[Array, "*batch 3"] | None,
    rotation_type: RotationType,
) -> Float[Array, "*batch N 3"]:
    """Apply global rotation and translation to points [..., N, 3]."""
    if rotation is not None:
        R = SO3.convert(rotation, src=rotation_type, dst="rotmat", xp=xp)
        points = (R @ points.mT).mT
    if translation is not None:
        points = points + translation[..., None, :]
    return points


def _apply_global_transform_to_rt(
    xp,
    R: Float[Array, "*batch J 3 3"],
    t: Float[Array, "*batch J 3"],
    rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    translation: Float[Array, "*batch 3"] | None,
    rotation_type: RotationType,
) -> tuple[Float[Array, "*batch J 3 3"], Float[Array, "*batch J 3"]]:
    """Apply global rotation and translation to R, t components."""
    if rotation is not None:
        R_global = SO3.convert(rotation, src=rotation_type, dst="rotmat", xp=xp)
        # Transform t: R_global @ t
        t = (R_global @ t.mT).mT
        # Transform R: R_global @ R (broadcast R_global over J dimension)
        R = R_global[..., None, :, :] @ R
    if translation is not None:
        t = t + translation[..., None, :]
    return R, t
