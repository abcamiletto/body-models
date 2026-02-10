"""Backend-agnostic ANNY computation using array_api_compat."""

from typing import Any

import numpy as np
from array_api_compat import get_namespace
from jaxtyping import Float
from nanomanifold import SO3

from .. import common
from .io import PHENOTYPE_VARIATIONS

Array = Any  # Generic array type (numpy, torch, jax)

# Coordinate transform constants (Z-up to Y-up)
COORD_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
COORD_TRANSLATION = np.array([0.0, 0.852, 0.0], dtype=np.float32)


def forward_vertices(
    # Model data
    template_vertices: Float[Array, "V 3"],
    blendshapes: Float[Array, "S V 3"],
    template_bone_heads: Float[Array, "J 3"],
    template_bone_tails: Float[Array, "J 3"],
    bone_heads_blendshapes: Float[Array, "S J 3"],
    bone_tails_blendshapes: Float[Array, "S J 3"],
    bone_rolls_rotmat: Float[Array, "J 3 3"],
    lbs_weights: Float[Array, "V J"],
    phenotype_mask: Float[Array, "S P"],
    anchors: dict[str, Float[Array, "A"]],
    kinematic_fronts: tuple[list[list[int]], list[list[int]]],
    coord_rotation: Float[Array, "3 3"],
    coord_translation: Float[Array, "3"],
    y_axis: Float[Array, "3"],
    degenerate_rotation: Float[Array, "3 3"],
    extrapolate_phenotypes: bool,
    # Inputs
    gender: Float[Array, "B"],
    age: Float[Array, "B"],
    muscle: Float[Array, "B"],
    weight: Float[Array, "B"],
    height: Float[Array, "B"],
    proportions: Float[Array, "B"],
    pose: Float[Array, "B J 3"],
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3]."""
    assert gender.ndim == 1
    assert age.ndim == 1
    assert muscle.ndim == 1
    assert weight.ndim == 1
    assert height.ndim == 1
    assert proportions.ndim == 1
    assert pose.ndim == 3 and pose.shape[2] == 3
    assert global_rotation is None or (global_rotation.ndim == 2 and global_rotation.shape[1] == 3)
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(gender)

    pose_T = _axis_angle_to_transform(xp, pose)
    coeffs, _, bone_transforms = _forward_core(
        xp=xp,
        template_bone_heads=template_bone_heads,
        template_bone_tails=template_bone_tails,
        bone_heads_blendshapes=bone_heads_blendshapes,
        bone_tails_blendshapes=bone_tails_blendshapes,
        bone_rolls_rotmat=bone_rolls_rotmat,
        phenotype_mask=phenotype_mask,
        anchors=anchors,
        kinematic_fronts=kinematic_fronts,
        y_axis=y_axis,
        degenerate_rotation=degenerate_rotation,
        extrapolate_phenotypes=extrapolate_phenotypes,
        gender=gender,
        age=age,
        muscle=muscle,
        weight=weight,
        height=height,
        proportions=proportions,
        pose_T=pose_T,
    )

    # Vertex blendshapes
    rest_verts = template_vertices + xp.einsum("bs,svd->bvd", coeffs, blendshapes)

    # Linear blend skinning
    R = bone_transforms[..., :3, :3]  # [B, J, 3, 3]
    t = bone_transforms[..., :3, 3]  # [B, J, 3]
    W_R = xp.einsum("vj,bjkl->bvkl", lbs_weights, R)  # [B, V, 3, 3]
    W_t = xp.einsum("vj,bjk->bvk", lbs_weights, t)  # [B, V, 3]
    vertices = xp.squeeze(W_R @ rest_verts[..., None], axis=-1) + W_t

    # Coordinate transform + global
    vertices = vertices @ coord_rotation.T + coord_translation
    if global_rotation is not None:
        global_rotation = xp.asarray(global_rotation, dtype=vertices.dtype)
        R_global = SO3.to_matrix(SO3.from_axis_angle(global_rotation, xp=xp), xp=xp)
        vertices = (R_global @ vertices.mT).mT
    if global_translation is not None:
        global_translation = xp.asarray(global_translation, dtype=vertices.dtype)
        vertices = vertices + global_translation[:, None]
    return vertices


def forward_skeleton(
    # Model data
    template_bone_heads: Float[Array, "J 3"],
    template_bone_tails: Float[Array, "J 3"],
    bone_heads_blendshapes: Float[Array, "S J 3"],
    bone_tails_blendshapes: Float[Array, "S J 3"],
    bone_rolls_rotmat: Float[Array, "J 3 3"],
    phenotype_mask: Float[Array, "S P"],
    anchors: dict[str, Float[Array, "A"]],
    kinematic_fronts: tuple[list[list[int]], list[list[int]]],
    coord_rotation: Float[Array, "3 3"],
    coord_translation: Float[Array, "3"],
    y_axis: Float[Array, "3"],
    degenerate_rotation: Float[Array, "3 3"],
    extrapolate_phenotypes: bool,
    # Inputs
    gender: Float[Array, "B"],
    age: Float[Array, "B"],
    muscle: Float[Array, "B"],
    weight: Float[Array, "B"],
    height: Float[Array, "B"],
    proportions: Float[Array, "B"],
    pose: Float[Array, "B J 3"],
    global_rotation: Float[Array, "B 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
    """Compute skeleton transforms [B, J, 4, 4]."""
    assert gender.ndim == 1
    assert age.ndim == 1
    assert muscle.ndim == 1
    assert weight.ndim == 1
    assert height.ndim == 1
    assert proportions.ndim == 1
    assert pose.ndim == 3 and pose.shape[2] == 3
    assert global_rotation is None or (global_rotation.ndim == 2 and global_rotation.shape[1] == 3)
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(gender)

    pose_T = _axis_angle_to_transform(xp, pose)
    _, bone_poses, _ = _forward_core(
        xp=xp,
        template_bone_heads=template_bone_heads,
        template_bone_tails=template_bone_tails,
        bone_heads_blendshapes=bone_heads_blendshapes,
        bone_tails_blendshapes=bone_tails_blendshapes,
        bone_rolls_rotmat=bone_rolls_rotmat,
        phenotype_mask=phenotype_mask,
        anchors=anchors,
        kinematic_fronts=kinematic_fronts,
        y_axis=y_axis,
        degenerate_rotation=degenerate_rotation,
        extrapolate_phenotypes=extrapolate_phenotypes,
        gender=gender,
        age=age,
        muscle=muscle,
        weight=weight,
        height=height,
        proportions=proportions,
        pose_T=pose_T,
    )

    # Coordinate transform
    B = bone_poses.shape[0]
    idx_R = (slice(None, 3), slice(None, 3))
    idx_t = (slice(None, 3), 3)
    coord_T = common.zeros_as(bone_poses, shape=(4, 4))
    coord_T = common.set(coord_T, idx_R, coord_rotation, xp=xp)
    coord_T = common.set(coord_T, idx_t, coord_translation, xp=xp)
    coord_T = common.set(coord_T, (3, 3), xp.asarray(1.0, dtype=bone_poses.dtype), xp=xp)
    transforms = coord_T @ bone_poses

    # Global transform
    if global_rotation is not None or global_translation is not None:
        # Create contiguous identity matrices (broadcast returns non-contiguous view)
        idx_R = (slice(None), slice(None, 3), slice(None, 3))
        idx_t = (slice(None), slice(None, 3), 3)
        G = common.eye_as(transforms, batch_dims=(B,))
        if global_rotation is not None:
            global_rotation = xp.asarray(global_rotation, dtype=transforms.dtype)
            R_global = SO3.to_matrix(SO3.from_axis_angle(global_rotation, xp=xp), xp=xp)
            G = common.set(G, idx_R, R_global, copy=False, xp=xp)
        if global_translation is not None:
            global_translation = xp.asarray(global_translation, dtype=transforms.dtype)
            G = common.set(G, idx_t, global_translation, copy=False, xp=xp)
        transforms = G[:, None] @ transforms
    return transforms


def _forward_core(
    xp,
    template_bone_heads: Float[Array, "J 3"],
    template_bone_tails: Float[Array, "J 3"],
    bone_heads_blendshapes: Float[Array, "S J 3"],
    bone_tails_blendshapes: Float[Array, "S J 3"],
    bone_rolls_rotmat: Float[Array, "J 3 3"],
    phenotype_mask: Float[Array, "S P"],
    anchors: dict[str, Float[Array, "A"]],
    kinematic_fronts: tuple[list[list[int]], list[list[int]]],
    y_axis: Float[Array, "3"],
    degenerate_rotation: Float[Array, "3 3"],
    extrapolate_phenotypes: bool,
    gender: Float[Array, "B"],
    age: Float[Array, "B"],
    muscle: Float[Array, "B"],
    weight: Float[Array, "B"],
    height: Float[Array, "B"],
    proportions: Float[Array, "B"],
    pose_T: Float[Array, "B J 4 4"],
) -> tuple[Float[Array, "B S"], Float[Array, "B J 4 4"], Float[Array, "B J 4 4"]]:
    """Core forward: returns (blendshape_coeffs, bone_poses, bone_transforms)."""
    # Phenotype -> blendshape coefficients
    coeffs = _phenotype_to_coeffs(
        xp=xp,
        phenotype_mask=phenotype_mask,
        anchors=anchors,
        extrapolate_phenotypes=extrapolate_phenotypes,
        gender=gender,
        age=age,
        muscle=muscle,
        weight=weight,
        height=height,
        proportions=proportions,
    )

    # Rest bone poses from blendshapes
    heads = template_bone_heads + xp.einsum("bs,sjd->bjd", coeffs, bone_heads_blendshapes)
    tails = template_bone_tails + xp.einsum("bs,sjd->bjd", coeffs, bone_tails_blendshapes)
    rest_poses = _bone_poses_from_heads_tails(xp, heads, tails, bone_rolls_rotmat, y_axis, degenerate_rotation)

    # Root parameterization
    root_rest = rest_poses[:, 0]
    base_T = _invert_transform(xp, root_rest)

    idx_R = (slice(None), slice(None, 3), slice(None, 3))
    root_rot = xp.zeros_like(root_rest)
    root_rot = common.set(root_rot, idx_R, root_rest[:, :3, :3], copy=False, xp=xp)
    root_rot = common.set(root_rot, (slice(None), 3, 3), xp.asarray(1.0, dtype=root_rot.dtype), copy=False, xp=xp)
    new_root = pose_T[:, 0] @ root_rot
    # copy=True handles clone/copy for NumPy/PyTorch, creates new array for JAX
    delta_T = common.set(pose_T, (slice(None), 0), new_root, copy=True, xp=xp)

    # Forward kinematics
    bone_poses, bone_transforms = _forward_kinematics(xp, kinematic_fronts, rest_poses, delta_T, base_T)
    return coeffs, bone_poses, bone_transforms


def _phenotype_to_coeffs(
    xp,
    phenotype_mask: Float[Array, "S P"],
    anchors: dict[str, Float[Array, "A"]],
    extrapolate_phenotypes: bool,
    gender: Float[Array, "B"],
    age: Float[Array, "B"],
    muscle: Float[Array, "B"],
    weight: Float[Array, "B"],
    height: Float[Array, "B"],
    proportions: Float[Array, "B"],
    cupsize: float = 0.5,
    firmness: float = 0.5,
    african: float = 0.5,
    asian: float = 0.5,
    caucasian: float = 0.5,
) -> Float[Array, "B S"]:
    """Convert phenotype parameters to blendshape coefficients."""
    dtype = phenotype_mask.dtype

    # Convert scalar defaults to batched arrays (user inputs are already tensors)
    cupsize = xp.full_like(gender, cupsize, dtype=dtype)
    firmness = xp.full_like(gender, firmness, dtype=dtype)

    # Interpolation weights for each phenotype
    weights = {}
    for name, val in [
        ("gender", gender),
        ("age", age),
        ("muscle", muscle),
        ("weight", weight),
        ("height", height),
        ("proportions", proportions),
        ("cupsize", cupsize),
        ("firmness", firmness),
    ]:
        anchor_arr = anchors[name]
        n_anchors = anchor_arr.shape[0]

        # searchsorted equivalent
        idx = xp.sum(xp.asarray(val[:, None] >= anchor_arr[None, :], dtype=xp.int32), axis=1)
        idx = xp.clip(idx, 1, n_anchors - 1)

        # Interpolation alpha
        idx_m1 = idx - 1
        # Gather anchor values at idx and idx-1
        alpha_num = val - anchor_arr[idx_m1]
        alpha_den = anchor_arr[idx] - anchor_arr[idx_m1]
        alpha = alpha_num / alpha_den
        if not extrapolate_phenotypes:
            alpha = xp.clip(alpha, 0, 1)

        # Build weight matrix
        w = common.zeros_as(val, shape=(val.shape[0], n_anchors))
        w = xp.asarray(w, dtype=dtype)
        # Scatter 1-alpha at idx-1 and alpha at idx
        batch_indices = xp.cumsum(xp.ones_like(val, dtype=xp.int32), axis=0) - 1
        w = common.set(w, (batch_indices, idx_m1), 1 - alpha, copy=True, xp=xp)
        w = common.set(w, (batch_indices, idx), alpha, copy=False, xp=xp)

        weights[name] = {k: w[:, i] for i, k in enumerate(PHENOTYPE_VARIATIONS[name])}

    # Race weights (normalized) - convert scalar defaults to batched arrays
    african = xp.full_like(gender, african, dtype=dtype)
    asian = xp.full_like(gender, asian, dtype=dtype)
    caucasian = xp.full_like(gender, caucasian, dtype=dtype)
    race = xp.stack([african, asian, caucasian], axis=1)
    race_sum = xp.sum(race, axis=1, keepdims=True)
    race_sum = xp.where(race_sum == 0, xp.asarray(1.0, dtype=dtype), race_sum)
    race = race / race_sum

    # Stack all phenotype weights
    all_weights = {k: v for d in weights.values() for k, v in d.items()}
    all_weights.update(african=race[:, 0], asian=race[:, 1], caucasian=race[:, 2])

    phen_list = []
    for vs in PHENOTYPE_VARIATIONS.values():
        for k in vs:
            phen_list.append(all_weights[k])
    phens = xp.stack(phen_list, axis=1)

    # Compute blendshape coefficients via masked product
    masked = phens[:, None, :] * phenotype_mask[None, :, :]
    return xp.prod(masked + (1 - phenotype_mask[None, :, :]), axis=-1)


def _bone_poses_from_heads_tails(
    xp,
    heads: Float[Array, "B J 3"],
    tails: Float[Array, "B J 3"],
    rolls: Float[Array, "J 3 3"],
    y_axis: Float[Array, "3"],
    degen_rot: Float[Array, "3 3"],
    eps: float = 0.1,
) -> Float[Array, "B J 4 4"]:
    """Compute bone poses from head/tail positions."""
    vec = tails - heads
    y = vec / xp.linalg.vector_norm(vec, axis=-1, keepdims=True)
    y_axis_expanded = xp.broadcast_to(y_axis, y.shape)
    cross = xp.linalg.cross(y, y_axis_expanded)
    dot = xp.sum(y * y_axis_expanded, axis=-1)
    cross_norm = xp.linalg.vector_norm(cross, axis=-1)

    axis = cross / cross_norm[..., None]
    angle = xp.atan2(cross_norm, dot)
    R = SO3.to_matrix(SO3.from_axis_angle(-angle[..., None] * axis, xp=xp), xp=xp)

    valid = (xp.abs(xp.sum(axis**2, axis=-1) - 1) < eps)[..., None, None]
    degen_expanded = xp.broadcast_to(degen_rot, R.shape)
    R = xp.where(valid, R, degen_expanded)
    R = R @ rolls

    B, J = R.shape[:2]
    dtype = R.dtype
    idx_R = (..., slice(None, 3), slice(None, 3))
    idx_t = (..., slice(None, 3), 3)
    H = common.zeros_as(R, shape=(B, J, 4, 4))
    H = common.set(H, idx_R, R, xp=xp)
    H = common.set(H, idx_t, heads, xp=xp)
    H = common.set(H, (..., 3, 3), xp.asarray(1.0, dtype=dtype), xp=xp)
    return H


def _invert_transform(xp, T: Float[Array, "*batch 4 4"]) -> Float[Array, "*batch 4 4"]:
    """Invert a 4x4 rigid transform."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_t = R.mT
    idx_R = (..., slice(None, 3), slice(None, 3))
    idx_t = (..., slice(None, 3), 3)
    inv = xp.zeros_like(T)
    inv = common.set(inv, idx_R, R_t, xp=xp)
    inv = common.set(inv, idx_t, -xp.squeeze(R_t @ t[..., None], axis=-1), xp=xp)
    inv = common.set(inv, (..., 3, 3), xp.asarray(1.0, dtype=T.dtype), xp=xp)
    return inv


def _forward_kinematics(
    xp,
    fronts: tuple[list[list[int]], list[list[int]]],
    rest_poses: Float[Array, "B J 4 4"],
    delta_T: Float[Array, "B J 4 4"],
    base_T: Float[Array, "B 4 4"],
) -> tuple[Float[Array, "B J 4 4"], Float[Array, "B J 4 4"]]:
    """Parallel forward kinematics (autograd-compatible)."""
    B, J = rest_poses.shape[:2]

    T = rest_poses @ delta_T
    rest_inv = _invert_transform(xp, rest_poses)

    poses: list[Float[Array, "B 4 4"] | None] = [None] * J
    transforms: list[Float[Array, "B 4 4"] | None] = [None] * J

    indices_list, parents_list = fronts
    for joint_ids, parent_ids in zip(indices_list, parents_list):
        roots = [(j, p) for j, p in zip(joint_ids, parent_ids) if p == -1]
        children_list = [(j, p) for j, p in zip(joint_ids, parent_ids) if p >= 0]

        if roots:
            root_ids = [j for j, _ in roots]
            root_poses = base_T[:, None] @ T[:, root_ids]
            root_transforms = root_poses @ rest_inv[:, root_ids]
            for idx, joint_id in enumerate(root_ids):
                poses[joint_id] = root_poses[:, idx]
                transforms[joint_id] = root_transforms[:, idx]

        if children_list:
            child_ids = [j for j, _ in children_list]
            parent_ids_list = [p for _, p in children_list]
            parent_transforms = xp.stack([transforms[p] for p in parent_ids_list], axis=1)
            child_poses = parent_transforms @ T[:, child_ids]
            child_transforms = child_poses @ rest_inv[:, child_ids]
            for idx, joint_id in enumerate(child_ids):
                poses[joint_id] = child_poses[:, idx]
                transforms[joint_id] = child_transforms[:, idx]

    poses_tensor = xp.stack(poses, axis=1)
    transforms_tensor = xp.stack(transforms, axis=1)
    return poses_tensor, transforms_tensor


def _axis_angle_to_transform(xp, pose: Float[Array, "B J 3"]) -> Float[Array, "B J 4 4"]:
    """Convert axis-angle pose to 4x4 transforms."""
    R = SO3.to_matrix(SO3.from_axis_angle(pose, xp=xp), xp=xp)
    B, J = R.shape[:2]
    dtype = R.dtype
    idx_R = (..., slice(None, 3), slice(None, 3))
    T = common.zeros_as(R, shape=(B, J, 4, 4))
    T = common.set(T, idx_R, R, xp=xp)
    T = common.set(T, (..., 3, 3), xp.asarray(1.0, dtype=dtype), xp=xp)
    return T


def from_native_args(pose: Float[Array, "B J 4 4"]) -> dict[str, Array]:
    """Convert native ANNY args (4x4 transforms) to API format (axis-angle).

    Args:
        pose: Per-joint 4x4 rotation transforms [B, J, 4, 4] in Z-up coords

    Returns:
        Dict with 'pose' as axis-angle [B, J, 3]
    """
    xp = get_namespace(pose)
    R = pose[..., :3, :3]
    axis_angle = SO3.to_axis_angle(SO3.from_matrix(R, xp=xp), xp=xp)
    return {"pose": axis_angle}


def to_native_outputs(
    vertices: Float[Array, "B V 3"],
    transforms: Float[Array, "B J 4 4"],
) -> dict[str, Array]:
    """Convert API outputs (Y-up) to native ANNY format (Z-up).

    Args:
        vertices: Mesh vertices [B, V, 3] in Y-up coords
        transforms: Joint transforms [B, J, 4, 4] in Y-up coords

    Returns:
        Dict with 'vertices' and 'bone_poses' in Z-up coords
    """
    xp = get_namespace(vertices)
    dtype = vertices.dtype

    coord_rot = xp.asarray(COORD_ROTATION, dtype=dtype)
    coord_trans = xp.asarray(COORD_TRANSLATION, dtype=dtype)

    # Inverse transform: Y-up -> Z-up
    # Forward was: v_yup = v_zup @ R.T + t
    # Inverse: v_zup = (v_yup - t) @ R
    native_verts = (vertices - coord_trans) @ coord_rot

    # For transforms: T_yup = coord @ T_zup
    # Inverse: T_zup = coord_inv @ T_yup
    idx_R = (slice(None, 3), slice(None, 3))
    idx_t = (slice(None, 3), 3)
    coord_T = common.zeros_as(vertices, shape=(4, 4))
    coord_T = common.set(coord_T, idx_R, coord_rot, xp=xp)
    coord_T = common.set(coord_T, idx_t, coord_trans, xp=xp)
    coord_T = common.set(coord_T, (3, 3), xp.asarray(1.0, dtype=dtype), xp=xp)
    coord_T_inv = _invert_transform(xp, coord_T)
    native_transforms = coord_T_inv @ transforms

    return {"vertices": native_verts, "bone_poses": native_transforms}
