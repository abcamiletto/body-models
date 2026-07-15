"""Backend-agnostic linear blend skinning operations."""

from typing import Any

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models.common import ops
from body_models.rotations import RotationType

Array = Any


def linear_blend_skinning(
    vertices: Float[Array, "*batch V 3"],
    transforms: Float[Array, "*batch J 4 4"],
    weights: Float[Array, "V J"],
    *,
    xp: Any = None,
) -> Float[Array, "*batch V 3"]:
    """Blend joint transforms and apply them to vertices."""
    if xp is None:
        xp = ops.get_namespace(vertices)

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
    xp: Any = None,
) -> Float[Array, "*batch V 3"]:
    """Apply linear blend skinning from compact per-vertex joint weights."""
    if xp is None:
        xp = ops.get_namespace(vertices)

    num_vertices, num_slots = joint_indices.shape
    flat_indices = joint_indices.reshape(-1)

    rotations = transforms[..., flat_indices, :3, :3]
    rotations = rotations.reshape(*transforms.shape[:-3], num_vertices, num_slots, 3, 3)
    translations = transforms[..., flat_indices, :3, 3]
    translations = translations.reshape(*transforms.shape[:-3], num_vertices, num_slots, 3)

    transformed = xp.einsum("...vkij,...vj->...vki", rotations, vertices) + translations
    return xp.sum(transformed * joint_weights[..., None], axis=-2)


def bind_relative_transforms(
    skeleton_transforms: Float[Array, "*batch J 4 4"],
    rest_joints: Float[Array, "*batch J 3"],
    *,
    xp: Any = None,
) -> Float[Array, "*batch J 4 4"]:
    """Convert world transforms to bind-relative skinning transforms."""
    if xp is None:
        xp = ops.get_namespace(skeleton_transforms)

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
    xp: Any = None,
) -> Float[Array, "*batch N 3"]:
    """Apply an optional global rotation and translation to points."""
    if rotation is None and translation is None:
        return points
    if xp is None:
        xp = ops.get_namespace(points)

    if rotation is not None:
        rotation_matrix = SO3.convert(rotation, src=rotation_type, dst="rotmat", xp=xp)
        points = (rotation_matrix @ points.mT).mT
    if translation is not None:
        points = points + translation[..., None, :]
    return points
