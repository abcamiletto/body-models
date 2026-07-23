"""Backend-agnostic linear blend skinning operations."""

from typing import Any

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models.common import ops
from body_models.common.kinematics import affine_transforms
from body_models.rotations import RotationType

Array = Any


def linear_blend_skinning(
    vertices: Float[Array, "*batch V 3"],
    transforms: Float[Array, "*batch J 4 4"],
    weights: Float[Array, "V J"],
    *,
    xp: Any,
) -> Float[Array, "*batch V 3"]:
    """Blend joint transforms and apply them to vertices."""
    rotations = transforms[..., :3, :3]
    translations = transforms[..., :3, 3]
    blended_rotations = xp.einsum("vj,...jkl->...vkl", weights, rotations)
    blended_translations = xp.einsum("vj,...jk->...vk", weights, translations)
    rotated = xp.squeeze(blended_rotations @ vertices[..., None], axis=-1)
    return rotated + blended_translations


def compact_linear_blend_skinning(
    vertices: Float[Array, "*batch V 3"],
    transforms: Float[Array, "*batch J 4 4"],
    *,
    joint_indices: Int[Array, "V K"],
    joint_weights: Float[Array, "V K"],
    xp: Any,
) -> Float[Array, "*batch V 3"]:
    """Apply linear blend skinning from compact per-vertex joint weights."""
    result = xp.zeros_like(vertices)
    transforms_by_joint = xp.moveaxis(transforms, -3, 0)
    for slot in range(joint_indices.shape[1]):
        indices = joint_indices[:, slot]
        valid = indices >= 0
        safe_indices = xp.maximum(indices, xp.zeros_like(indices))
        vertex_transforms = xp.moveaxis(transforms_by_joint[safe_indices], 0, -3)
        rotations = vertex_transforms[..., :3, :3]
        translations = vertex_transforms[..., :3, 3]
        transformed = xp.einsum("...vij,...vj->...vi", rotations, vertices) + translations
        weights = joint_weights[:, slot] * valid
        result = result + transformed * weights[:, None]
    return result


def bind_relative_transforms(
    skeleton_transforms: Float[Array, "*batch J 4 4"],
    rest_joints: Float[Array, "*batch J 3"],
    *,
    xp: Any,
) -> Float[Array, "*batch J 4 4"]:
    """Convert world transforms to bind-relative skinning transforms."""
    rotations = skeleton_transforms[..., :3, :3]
    translations = skeleton_transforms[..., :3, 3]
    bind_translations = translations - xp.squeeze(rotations @ rest_joints[..., None], axis=-1)
    return ops.set(
        skeleton_transforms,
        (..., slice(None, 3), 3),
        bind_translations,
        xp=xp,
    )


def apply_global_transform(
    points: Float[Array, "*batch N 3"],
    rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    translation: Float[Array, "*batch 3"] | None,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any,
) -> Float[Array, "*batch N 3"]:
    """Apply an optional global rotation and translation to points."""
    if rotation is None and translation is None:
        return points
    if rotation is not None:
        rotation_matrix = SO3.convert(rotation, src=rotation_type, dst="rotmat", xp=xp)
        points = (rotation_matrix @ points.mT).mT
    if translation is not None:
        points = points + translation[..., None, :]
    return points


def transform_skeleton(
    transforms: Float[Array, "*batch J 4 4"],
    rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    translation: Float[Array, "*batch 3"] | None,
    rotation_type: RotationType = "axis_angle",
    joint_indices: list[int] | None = None,
    *,
    xp: Any,
) -> Float[Array, "*batch J 4 4"]:
    """Select joints and apply an optional global transform to a skeleton."""
    if translation is not None and (translation.ndim < 1 or translation.shape[-1] != 3):
        raise ValueError("translation must have shape [..., 3]")
    if joint_indices is not None:
        joint_indices = [int(joint) for joint in joint_indices]
        num_joints = transforms.shape[-3]
        if any(joint < 0 or joint >= num_joints for joint in joint_indices):
            raise IndexError(f"joint_indices must be in [0, {num_joints})")
        transforms = transforms[..., joint_indices, :, :]

    if rotation is None:
        if translation is None:
            return transforms
        positions = transforms[..., :3, 3] + translation[..., None, :]
        return ops.set(transforms, (..., slice(None, 3), 3), positions, xp=xp)

    rotations = transforms[..., :3, :3]
    positions = transforms[..., :3, 3]
    global_rotation = SO3.convert(rotation, src=rotation_type, dst="rotmat", xp=xp)
    positions = (global_rotation @ positions.mT).mT
    rotations = global_rotation[..., None, :, :] @ rotations
    if translation is not None:
        positions = positions + translation[..., None, :]

    return affine_transforms(rotations, positions, xp=xp)
