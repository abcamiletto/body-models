"""Backend-independent ANNY pose and identity preparation."""

from typing import Any, TypedDict

from jaxtyping import Float
from nanomanifold import SO3

from body_models import common
from body_models.bodies.anny.io import PHENOTYPE_VARIATIONS
from body_models.rotations import RotationType

Array = Any
Front = tuple[list[int], list[int]]


class AnnySkeletonIdentity(TypedDict):
    """Phenotype-dependent joint state needed to pose the ANNY skeleton."""

    rest_skeleton_transforms: Float[Array, "*batch J 4 4"]


class AnnyIdentity(AnnySkeletonIdentity):
    """Complete phenotype-dependent ANNY mesh state."""

    rest_vertices: Float[Array, "*batch V 3"]


class AnnyPreparedPose(TypedDict):
    """Complete pose-dependent ANNY mesh state."""

    skeleton_transforms: Float[Array, "*batch J 4 4"]
    skinning_transforms: Float[Array, "*batch J 4 4"]


def shape_vertices(
    template_vertices: Float[Array, "V 3"],
    blendshapes: Float[Array, "S V 3"],
    phenotype_mask: Float[Array, "S P"],
    anchors: Any,
    shape: Float[Array, "*batch 6"],
    *,
    extrapolate_phenotypes: bool = False,
    xp: Any,
) -> Float[Array, "*batch V 3"]:
    """Evaluate ANNY's phenotype-controlled rest surface."""
    coefficients = _phenotype_to_coeffs(
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
    return template_vertices + xp.einsum("...s,svd->...vd", coefficients, blendshapes)


def prepare_pose(
    kinematic_fronts: list[Front],
    pose: Float[Array, "*batch J N"] | Float[Array, "*batch J 3 3"],
    rotation_type: RotationType,
    *,
    rest_skeleton_transforms: Float[Array, "*batch J 4 4"],
    xp: Any,
) -> AnnyPreparedPose:
    """Prepare ANNY skeleton and bind-relative skinning transforms."""
    pose_transforms = _pose_to_transform(xp, pose, rotation_type)
    skeleton_transforms, skinning_transforms = _forward_core(
        xp=xp,
        kinematic_fronts=kinematic_fronts,
        rest_skeleton_transforms=rest_skeleton_transforms,
        pose_transforms=pose_transforms,
        skip_skinning=False,
    )
    assert skinning_transforms is not None
    return {
        "skeleton_transforms": skeleton_transforms,
        "skinning_transforms": skinning_transforms,
    }


def prepare_skeleton(
    kinematic_fronts: list[Front],
    pose: Float[Array, "*batch J N"] | Float[Array, "*batch J 3 3"],
    rotation_type: RotationType,
    *,
    rest_skeleton_transforms: Float[Array, "*batch J 4 4"],
    xp: Any,
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed ANNY joint transforms."""
    pose_transforms = _pose_to_transform(xp, pose, rotation_type)
    skeleton, _ = _forward_core(
        xp=xp,
        kinematic_fronts=kinematic_fronts,
        rest_skeleton_transforms=rest_skeleton_transforms,
        pose_transforms=pose_transforms,
        skip_skinning=True,
    )
    return skeleton


def _forward_core(
    xp: Any,
    kinematic_fronts: list[Front],
    rest_skeleton_transforms: Float[Array, "*batch J 4 4"],
    pose_transforms: Float[Array, "*batch J 4 4"],
    skip_skinning: bool,
) -> tuple[Float[Array, "*batch J 4 4"], Float[Array, "*batch J 4 4"] | None]:
    rest_poses = rest_skeleton_transforms
    root_rest = rest_poses[..., 0, :, :]
    base_transform = common.invert_rigid_transforms(root_rest, xp=xp)

    root_rotation = xp.zeros_like(root_rest)
    root_rotation = common.set(
        root_rotation,
        (..., slice(None, 3), slice(None, 3)),
        root_rest[..., :3, :3],
        copy=False,
        xp=xp,
    )
    root_rotation = common.set(
        root_rotation,
        (..., 3, 3),
        xp.asarray(1.0, dtype=root_rotation.dtype),
        copy=False,
        xp=xp,
    )
    posed_root = pose_transforms[..., 0, :, :] @ root_rotation
    delta_transforms = common.set(
        pose_transforms,
        (..., 0, slice(None), slice(None)),
        posed_root,
        copy=True,
        xp=xp,
    )
    return _forward_kinematics(
        xp,
        kinematic_fronts,
        rest_poses,
        delta_transforms,
        base_transform,
        skip_skinning=skip_skinning,
    )


def prepare_identity(
    *,
    xp: Any,
    template_vertices: Float[Array, "V 3"],
    blendshapes: Float[Array, "S V 3"],
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
) -> AnnyIdentity:
    """Prepare phenotype-dependent ANNY skeleton and vertices."""
    coefficients, rest_skeleton = _prepare_identity_state(
        xp=xp,
        template_bone_heads=template_bone_heads,
        template_bone_tails=template_bone_tails,
        bone_heads_blendshapes=bone_heads_blendshapes,
        bone_tails_blendshapes=bone_tails_blendshapes,
        bone_rolls_rotmat=bone_rolls_rotmat,
        phenotype_mask=phenotype_mask,
        anchors=anchors,
        y_axis=y_axis,
        degenerate_rotation=degenerate_rotation,
        extrapolate_phenotypes=extrapolate_phenotypes,
        shape=shape,
    )
    return {
        "rest_skeleton_transforms": rest_skeleton,
        "rest_vertices": template_vertices + xp.einsum("...s,svd->...vd", coefficients, blendshapes),
    }


def prepare_skeleton_identity(
    *,
    xp: Any,
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
) -> AnnySkeletonIdentity:
    """Prepare only phenotype-dependent ANNY joint state."""
    _, rest_skeleton = _prepare_identity_state(
        xp=xp,
        template_bone_heads=template_bone_heads,
        template_bone_tails=template_bone_tails,
        bone_heads_blendshapes=bone_heads_blendshapes,
        bone_tails_blendshapes=bone_tails_blendshapes,
        bone_rolls_rotmat=bone_rolls_rotmat,
        phenotype_mask=phenotype_mask,
        anchors=anchors,
        y_axis=y_axis,
        degenerate_rotation=degenerate_rotation,
        extrapolate_phenotypes=extrapolate_phenotypes,
        shape=shape,
    )
    return {"rest_skeleton_transforms": rest_skeleton}


def _prepare_identity_state(
    *,
    xp: Any,
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
) -> tuple[Float[Array, "*batch S"], Float[Array, "*batch J 4 4"]]:
    if shape.shape[-1] != 6:
        raise ValueError(f"shape must have shape [..., 6], got {tuple(shape.shape)}")
    coefficients = _phenotype_to_coeffs(
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
    heads = template_bone_heads + xp.einsum("...s,sjd->...jd", coefficients, bone_heads_blendshapes)
    tails = template_bone_tails + xp.einsum("...s,sjd->...jd", coefficients, bone_tails_blendshapes)
    rest_skeleton = _skeleton_transforms_from_heads_tails(
        xp, heads, tails, bone_rolls_rotmat, y_axis, degenerate_rotation
    )
    return coefficients, rest_skeleton


def _phenotype_to_coeffs(
    xp: Any,
    phenotype_mask: Float[Array, "S P"],
    anchors: Any,
    extrapolate_phenotypes: bool,
    gender: Float[Array, "*batch"],
    age: Float[Array, "*batch"],
    muscle: Float[Array, "*batch"],
    weight: Float[Array, "*batch"],
    height: Float[Array, "*batch"],
    proportions: Float[Array, "*batch"],
) -> Float[Array, "*batch S"]:
    """Convert six phenotype controls to ANNY blendshape coefficients."""
    dtype = phenotype_mask.dtype
    values = {
        "gender": gender,
        "age": age,
        "muscle": muscle,
        "weight": weight,
        "height": height,
        "proportions": proportions,
        "cupsize": xp.full_like(gender, 0.5, dtype=dtype),
        "firmness": xp.full_like(gender, 0.5, dtype=dtype),
    }

    weights = {}
    for name, value in values.items():
        anchor_values = getattr(anchors, name)
        num_anchors = anchor_values.shape[0]
        index = xp.sum(xp.asarray(value[..., None] >= anchor_values, dtype=xp.int32), axis=-1)
        index = xp.clip(index, 1, num_anchors - 1)
        previous = index - 1
        alpha = (value - anchor_values[previous]) / (anchor_values[index] - anchor_values[previous])
        if not extrapolate_phenotypes:
            alpha = xp.clip(alpha, 0, 1)

        anchor_indices = xp.cumsum(xp.ones_like(anchor_values, dtype=index.dtype), axis=0) - 1
        interpolation = xp.where(anchor_indices == previous[..., None], 1 - alpha[..., None], 0)
        interpolation += xp.where(anchor_indices == index[..., None], alpha[..., None], 0)
        weights.update(
            {label: interpolation[..., position] for position, label in enumerate(PHENOTYPE_VARIATIONS[name])}
        )

    race = xp.full((*gender.shape, 3), 1 / 3, dtype=dtype)
    weights.update(african=race[..., 0], asian=race[..., 1], caucasian=race[..., 2])
    phenotype_weights = xp.stack(
        [weights[label] for labels in PHENOTYPE_VARIATIONS.values() for label in labels],
        axis=-1,
    )
    masked = phenotype_weights[..., None, :] * phenotype_mask
    return xp.prod(masked + (1 - phenotype_mask), axis=-1)


def _skeleton_transforms_from_heads_tails(
    xp: Any,
    heads: Float[Array, "*batch J 3"],
    tails: Float[Array, "*batch J 3"],
    rolls: Float[Array, "J 3 3"],
    y_axis: Float[Array, "3"],
    degenerate_rotation: Float[Array, "3 3"],
    eps: float = 0.1,
) -> Float[Array, "*batch J 4 4"]:
    vectors = tails - heads
    vector_norms = xp.linalg.vector_norm(vectors, axis=-1, keepdims=True)
    directions = vectors / xp.where(vector_norms > 0, vector_norms, xp.ones_like(vector_norms))
    reference = xp.broadcast_to(y_axis, directions.shape)
    cross = xp.linalg.cross(directions, reference)
    dot = xp.sum(directions * reference, axis=-1)
    cross_norm = xp.linalg.vector_norm(cross, axis=-1)
    axis = cross / xp.where(cross_norm > 0, cross_norm, xp.ones_like(cross_norm))[..., None]
    angle = xp.atan2(cross_norm, dot)
    rotations = SO3.conversions.from_axis_angle_to_rotmat(-angle[..., None] * axis, xp=xp)

    valid = (xp.abs(xp.sum(axis**2, axis=-1) - 1) < eps)[..., None, None]
    rotations = xp.where(valid, rotations, xp.broadcast_to(degenerate_rotation, rotations.shape)) @ rolls
    return common.affine_transforms(rotations, heads, xp=xp)


def _forward_kinematics(
    xp: Any,
    fronts: list[Front],
    rest_poses: Float[Array, "*batch J 4 4"],
    delta_transforms: Float[Array, "*batch J 4 4"],
    base_transform: Float[Array, "*batch 4 4"],
    *,
    skip_skinning: bool,
) -> tuple[Float[Array, "*batch J 4 4"], Float[Array, "*batch J 4 4"] | None]:
    num_joints = rest_poses.shape[-3]
    local_transforms = rest_poses @ delta_transforms
    inverse_rest = common.invert_rigid_transforms(rest_poses, xp=xp)
    poses: list[Any] = [None] * num_joints
    skinning_transforms: list[Any] = [None] * num_joints

    for joint_ids, parent_ids in fronts:
        roots = [(joint, parent) for joint, parent in zip(joint_ids, parent_ids) if parent == -1]
        children = [(joint, parent) for joint, parent in zip(joint_ids, parent_ids) if parent >= 0]
        if roots:
            root_ids = [joint for joint, _ in roots]
            root_poses = base_transform[..., None, :, :] @ local_transforms[..., root_ids, :, :]
            root_skinning = root_poses @ inverse_rest[..., root_ids, :, :]
            for index, joint in enumerate(root_ids):
                poses[joint] = root_poses[..., index, :, :]
                skinning_transforms[joint] = root_skinning[..., index, :, :]
        if children:
            child_ids = [joint for joint, _ in children]
            parent_skinning = xp.stack([skinning_transforms[parent] for _, parent in children], axis=-3)
            child_poses = parent_skinning @ local_transforms[..., child_ids, :, :]
            child_inverse_rest = inverse_rest[..., child_ids, :, :][..., None, :, :]
            child_skinning = xp.sum(child_poses[..., :, None] * child_inverse_rest, axis=-2)
            for index, joint in enumerate(child_ids):
                poses[joint] = child_poses[..., index, :, :]
                skinning_transforms[joint] = child_skinning[..., index, :, :]

    posed = xp.stack(poses, axis=-3)
    skinned = None if skip_skinning else xp.stack(skinning_transforms, axis=-3)
    return posed, skinned


def _pose_to_transform(
    xp: Any,
    pose: Float[Array, "*batch J N"] | Float[Array, "*batch J 3 3"],
    rotation_type: RotationType,
) -> Float[Array, "*batch J 4 4"]:
    rotations = SO3.convert(pose, src=rotation_type, dst="rotmat", xp=xp)
    return common.affine_transforms(rotations, xp=xp)


__all__ = ["AnnyIdentity", "AnnyPreparedPose", "prepare_identity", "prepare_pose", "shape_vertices"]
