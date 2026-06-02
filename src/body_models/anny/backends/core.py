"""Backend-agnostic ANNY computation."""

from typing import Any, NotRequired, TypedDict

from jaxtyping import Float
from nanomanifold import SO3

from ... import common
from ...common import get_namespace
from ...rotations import RotationType
from ..io import PHENOTYPE_VARIATIONS

Array = Any  # Generic array type (numpy, torch, jax)
Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).
IDENTITY_LABELS = ("gender", "age", "muscle", "weight", "height", "proportions")


class AnnyIdentity(TypedDict):
    """Phenotype-dependent ANNY state returned by ``prepare_identity``."""

    rest_skeleton_transforms: Float[Array, "*batch J 4 4"]
    rest_vertices: NotRequired[Float[Array, "*batch V 3"]]


class AnnyPreparedPose(TypedDict):
    """Pose-dependent ANNY state returned by ``prepare_pose``."""

    skeleton_transforms: Float[Array, "*batch J 4 4"]
    skinning_transforms: NotRequired[Float[Array, "*batch J 4 4"]]
    pose_offsets: NotRequired[Float[Array, "*batch V 3"]]


def forward_vertices(
    # Model data
    lbs_weights: Float[Array, "V J"],
    rest_vertices: Float[Array, "*batch V 3"],
    skinning_transforms: Float[Array, "*batch J 4 4"],
    global_translation: Float[Array, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    xp: Any = None,
) -> Float[Array, "*batch V 3"]:
    """Compute mesh vertices [B, V, 3]."""
    if xp is None:
        xp = get_namespace(rest_vertices)

    rest_verts = rest_vertices
    lbs = lbs_weights if vertex_indices is None else lbs_weights[xp.asarray(vertex_indices)]
    if vertex_indices is not None:
        rest_verts = rest_verts[..., xp.asarray(vertex_indices), :]
    vertices = linear_blend_skinning(xp, rest_verts, skinning_transforms, lbs)
    return apply_global_transform(xp, vertices, global_translation)


def prepare_pose(
    # Model data
    kinematic_fronts: list[Front],
    pose: Float[Array, "*batch J N"] | Float[Array, "*batch J 3 3"],
    rotation_type: RotationType = "axis_angle",
    *,
    rest_skeleton_transforms: Float[Array, "*batch J 4 4"],
    rest_vertices: Float[Array, "*batch V 3"] | None = None,
    skip_vertices: bool = False,
    xp: Any = None,
) -> AnnyPreparedPose:
    """Precompute pose-dependent ANNY state for repeated forward passes."""
    if xp is None:
        xp = get_namespace(pose)

    pose_T = _pose_to_transform(xp, pose, rotation_type)
    skeleton_transforms, skinning_transforms = _forward_core(
        xp=xp,
        kinematic_fronts=kinematic_fronts,
        rest_skeleton_transforms=rest_skeleton_transforms,
        pose_T=pose_T,
        skip_transforms=skip_vertices,
    )
    prepared_pose: AnnyPreparedPose = {"skeleton_transforms": skeleton_transforms}
    if skip_vertices:
        return prepared_pose
    assert rest_vertices is not None
    assert skinning_transforms is not None
    prepared_pose["skinning_transforms"] = skinning_transforms
    prepared_pose["pose_offsets"] = common.zeros_as(rest_vertices, shape=rest_vertices.shape, xp=xp)
    return prepared_pose


def linear_blend_skinning(
    xp,
    rest_verts: Float[Array, "*batch V 3"],
    skinning_transforms: Float[Array, "*batch J 4 4"],
    lbs_weights: Float[Array, "V J"],
) -> Float[Array, "*batch V 3"]:
    R = skinning_transforms[..., :3, :3]
    t = skinning_transforms[..., :3, 3]
    W_R = xp.einsum("vj,...jkl->...vkl", lbs_weights, R)
    W_t = xp.einsum("vj,...jk->...vk", lbs_weights, t)
    rotated = xp.squeeze(W_R @ rest_verts[..., None], axis=-1)
    return rotated + W_t


def apply_global_transform(
    xp,
    vertices: Float[Array, "*batch V 3"],
    global_translation: Float[Array, "B 3"] | None = None,
) -> Float[Array, "B V 3"]:
    if global_translation is not None:
        global_translation = xp.asarray(global_translation, dtype=vertices.dtype)
        vertices = vertices + global_translation[..., None, :]
    return vertices


def identity_shape(
    template_vertices: Float[Array, "V 3"],
    blendshapes: Float[Array, "S V 3"],
    phenotype_mask: Float[Array, "S P"],
    anchors: Any,
    shape: Float[Array, "*batch 6"],
    *,
    extrapolate_phenotypes: bool = False,
    xp: Any = None,
) -> Float[Array, "*batch V 3"]:
    """Compute ANNY rest vertices from the six public phenotype controls."""
    if xp is None:
        xp = get_namespace(shape)

    coeffs = _phenotype_to_coeffs(
        xp=xp,
        phenotype_mask=phenotype_mask,
        anchors=anchors,
        extrapolate_phenotypes=extrapolate_phenotypes,
        gender=shape[..., 0],
        age=shape[..., 1],
        muscle=shape[..., 2],
        weight=shape[..., 3],
        height=shape[..., 4],
        proportions=shape[..., 5],
    )
    return template_vertices + xp.einsum("...s,svd->...vd", coeffs, blendshapes)


def forward_skeleton(
    # Model data
    skeleton_transforms: Float[Array, "*batch J 4 4"],
    global_translation: Float[Array, "*batch 3"] | None = None,
    joint_indices: list[int] | None = None,
    xp: Any = None,
) -> Float[Array, "*batch J 4 4"]:
    """Compute skeleton transforms [B, J, 4, 4]."""
    if xp is None:
        xp = get_namespace(skeleton_transforms)
    transforms = skeleton_transforms
    if joint_indices is not None:
        joint_indices = [int(joint) for joint in joint_indices]
        num_joints = skeleton_transforms.shape[-3]
        if any(joint < 0 or joint >= num_joints for joint in joint_indices):
            raise IndexError(f"joint_indices must be in [0, {num_joints})")
        transforms = transforms[..., joint_indices, :, :]

    # Global transform
    if global_translation is not None:
        # Create contiguous identity matrices (broadcast returns non-contiguous view)
        batch_shape = transforms.shape[:-3]
        G = common.eye_as(transforms, batch_dims=batch_shape, xp=xp)
        global_translation = xp.asarray(global_translation, dtype=transforms.dtype)
        G = common.set(G, (..., slice(None, 3), 3), global_translation, copy=False, xp=xp)
        transforms = G[..., None, :, :] @ transforms
    return transforms


def _forward_core(
    xp,
    kinematic_fronts: list[Front],
    rest_skeleton_transforms: Float[Array, "*batch J 4 4"],
    pose_T: Float[Array, "*batch J 4 4"],
    skip_transforms: bool = False,
    joint_indices: list[int] | None = None,
) -> tuple[Float[Array, "B J 4 4"], Float[Array, "B J 4 4"]]:
    """Core forward: returns (skeleton_transforms, skinning_transforms)."""
    rest_poses = rest_skeleton_transforms

    # Root parameterization
    root_rest = rest_poses[..., 0, :, :]
    base_T = _invert_transform(xp, root_rest)

    root_rot = xp.zeros_like(root_rest)
    root_rot = common.set(root_rot, (..., slice(None, 3), slice(None, 3)), root_rest[..., :3, :3], copy=False, xp=xp)
    root_rot = common.set(root_rot, (..., 3, 3), xp.asarray(1.0, dtype=root_rot.dtype), copy=False, xp=xp)
    new_root = pose_T[..., 0, :, :] @ root_rot
    # copy=True handles clone/copy for NumPy/PyTorch, creates new array for JAX
    delta_T = common.set(pose_T, (..., 0, slice(None), slice(None)), new_root, copy=True, xp=xp)

    # Forward kinematics
    skeleton_transforms, skinning_transforms = _forward_kinematics(
        xp,
        kinematic_fronts,
        rest_poses,
        delta_T,
        base_T,
        skip_transforms=skip_transforms,
        joint_indices=joint_indices,
    )
    return skeleton_transforms, skinning_transforms


def prepare_identity(
    *,
    xp,
    template_vertices: Float[Array, "V 3"] | None,
    blendshapes: Float[Array, "S V 3"] | None,
    template_bone_heads: Float[Array, "J 3"],
    template_bone_tails: Float[Array, "J 3"],
    bone_heads_blendshapes: Float[Array, "S J 3"],
    bone_tails_blendshapes: Float[Array, "S J 3"],
    bone_rolls_rotmat: Float[Array, "J 3 3"],
    phenotype_mask: Float[Array, "S P"],
    anchors: Any,
    y_axis: Float[Array, "3"],
    degenerate_rotation: Float[Array, "3 3"],
    extrapolate_phenotypes: bool,
    shape: Float[Array, "*batch 6"],
    skip_vertices: bool = False,
) -> AnnyIdentity:
    """Precompute phenotype-dependent ANNY state for repeated forward passes."""
    coeffs = _phenotype_to_coeffs(
        xp=xp,
        phenotype_mask=phenotype_mask,
        anchors=anchors,
        extrapolate_phenotypes=extrapolate_phenotypes,
        gender=shape[..., 0],
        age=shape[..., 1],
        muscle=shape[..., 2],
        weight=shape[..., 3],
        height=shape[..., 4],
        proportions=shape[..., 5],
    )
    heads = template_bone_heads + xp.einsum("...s,sjd->...jd", coeffs, bone_heads_blendshapes)
    tails = template_bone_tails + xp.einsum("...s,sjd->...jd", coeffs, bone_tails_blendshapes)
    identity: AnnyIdentity = {
        "rest_skeleton_transforms": _skeleton_transforms_from_heads_tails(
            xp, heads, tails, bone_rolls_rotmat, y_axis, degenerate_rotation
        )
    }
    if not skip_vertices:
        assert template_vertices is not None and blendshapes is not None
        identity["rest_vertices"] = template_vertices + xp.einsum("...s,svd->...vd", coeffs, blendshapes)
    return identity


def _phenotype_to_coeffs(
    xp,
    phenotype_mask: Float[Array, "S P"],
    anchors: dict[str, Float[Array, "A"]],
    extrapolate_phenotypes: bool,
    gender: Float[Array, "*batch"],
    age: Float[Array, "*batch"],
    muscle: Float[Array, "*batch"],
    weight: Float[Array, "*batch"],
    height: Float[Array, "*batch"],
    proportions: Float[Array, "*batch"],
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
        anchor_arr = getattr(anchors, name)
        n_anchors = anchor_arr.shape[0]

        # searchsorted equivalent
        idx = xp.sum(xp.asarray(val[..., None] >= anchor_arr, dtype=xp.int32), axis=-1)
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
        anchor_indices = xp.cumsum(xp.ones_like(anchor_arr, dtype=idx.dtype), axis=0) - 1
        w = xp.where(anchor_indices == idx_m1[..., None], 1 - alpha[..., None], 0)
        w = w + xp.where(anchor_indices == idx[..., None], alpha[..., None], 0)

        weights[name] = {k: w[..., i] for i, k in enumerate(PHENOTYPE_VARIATIONS[name])}

    # Race weights (normalized) - convert scalar defaults to batched arrays
    african = xp.full_like(gender, african, dtype=dtype)
    asian = xp.full_like(gender, asian, dtype=dtype)
    caucasian = xp.full_like(gender, caucasian, dtype=dtype)
    race = xp.stack([african, asian, caucasian], axis=-1)
    race_sum = xp.sum(race, axis=-1, keepdims=True)
    race_sum = xp.where(race_sum == 0, xp.ones_like(race_sum), race_sum)
    race = race / race_sum

    # Stack all phenotype weights
    all_weights = {k: v for d in weights.values() for k, v in d.items()}
    all_weights.update(african=race[..., 0], asian=race[..., 1], caucasian=race[..., 2])

    phen_list = []
    for vs in PHENOTYPE_VARIATIONS.values():
        for k in vs:
            phen_list.append(all_weights[k])
    phens = xp.stack(phen_list, axis=-1)

    # Compute blendshape coefficients via masked product
    masked = phens[..., None, :] * phenotype_mask
    return xp.prod(masked + (1 - phenotype_mask), axis=-1)


def _skeleton_transforms_from_heads_tails(
    xp,
    heads: Float[Array, "*batch J 3"],
    tails: Float[Array, "*batch J 3"],
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
    R = SO3.conversions.from_axis_angle_to_rotmat(-angle[..., None] * axis, xp=xp)

    valid = (xp.abs(xp.sum(axis**2, axis=-1) - 1) < eps)[..., None, None]
    degen_expanded = xp.broadcast_to(degen_rot, R.shape)
    R = xp.where(valid, R, degen_expanded)
    R = R @ rolls

    batch_shape = R.shape[:-3]
    J = R.shape[-3]
    dtype = R.dtype
    idx_R = (..., slice(None, 3), slice(None, 3))
    idx_t = (..., slice(None, 3), 3)
    H = common.zeros_as(R, shape=(*batch_shape, J, 4, 4), xp=xp)
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
    t_inv = -xp.squeeze(R_t @ t[..., None], axis=-1)
    inv = common.set(inv, idx_t, t_inv, xp=xp)
    inv = common.set(inv, (..., 3, 3), xp.asarray(1.0, dtype=T.dtype), xp=xp)
    return inv


def _forward_kinematics(
    xp,
    fronts: list[Front],
    rest_poses: Float[Array, "*batch J 4 4"],
    delta_T: Float[Array, "*batch J 4 4"],
    base_T: Float[Array, "*batch 4 4"],
    skip_transforms: bool = False,
    joint_indices: list[int] | None = None,
) -> tuple[Float[Array, "B J 4 4"], Float[Array, "B J 4 4"] | None]:
    """Parallel forward kinematics (autograd-compatible)."""
    J = rest_poses.shape[-3]

    T = rest_poses @ delta_T
    rest_inv = _invert_transform(xp, rest_poses)

    poses: list[Float[Array, "*batch 4 4"] | None] = [None] * J
    transforms: list[Float[Array, "*batch 4 4"] | None] = [None] * J

    for joint_ids, parent_ids in fronts:
        roots = [(j, p) for j, p in zip(joint_ids, parent_ids) if p == -1]
        children_list = [(j, p) for j, p in zip(joint_ids, parent_ids) if p >= 0]

        if roots:
            root_ids = [j for j, _ in roots]
            root_poses = base_T[..., None, :, :] @ T[..., root_ids, :, :]
            root_transforms = root_poses @ rest_inv[..., root_ids, :, :]
            for idx, joint_id in enumerate(root_ids):
                poses[joint_id] = root_poses[..., idx, :, :]
                transforms[joint_id] = root_transforms[..., idx, :, :]

        if children_list:
            child_ids = [j for j, _ in children_list]
            parent_ids_list = [p for _, p in children_list]
            parent_transforms = xp.stack([transforms[p] for p in parent_ids_list], axis=-3)
            child_poses = parent_transforms @ T[..., child_ids, :, :]
            child_rest_inv = rest_inv[..., child_ids, :, :][..., None, :, :]
            child_transforms = xp.sum(child_poses[..., :, None] * child_rest_inv, axis=-2)
            for idx, joint_id in enumerate(child_ids):
                poses[joint_id] = child_poses[..., idx, :, :]
                transforms[joint_id] = child_transforms[..., idx, :, :]

    if joint_indices is None:
        poses_tensor = xp.stack(poses, axis=-3)
        transforms_tensor = None if skip_transforms else xp.stack(transforms, axis=-3)
    else:
        poses_tensor = xp.stack([poses[j] for j in joint_indices], axis=-3)
        transforms_tensor = None if skip_transforms else xp.stack([transforms[j] for j in joint_indices], axis=-3)
    return poses_tensor, transforms_tensor


def _pose_to_transform(
    xp,
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    rotation_type: RotationType,
) -> Float[Array, "B J 4 4"]:
    """Convert per-joint rotations to 4x4 transforms."""
    R = SO3.convert(pose, src=rotation_type, dst="rotmat", xp=xp)
    batch_shape = R.shape[:-3]
    J = R.shape[-3]
    dtype = R.dtype
    idx_R = (..., slice(None, 3), slice(None, 3))
    T = common.zeros_as(R, shape=(*batch_shape, J, 4, 4), xp=xp)
    T = common.set(T, idx_R, R, xp=xp)
    T = common.set(T, (..., 3, 3), xp.asarray(1.0, dtype=dtype), xp=xp)
    return T
