"""Backend-agnostic linear deformation primitives."""

from typing import Any

from jaxtyping import Float

from body_models.common import ops

Array = Any


def blend_shapes(
    mean: Float[Array, "V D"],
    directions: Float[Array, "V D C"],
    coefficients: Float[Array, "*batch C"],
    *,
    xp: Any,
) -> Float[Array, "*batch V D"]:
    """Apply a linear blend-shape basis stored along its final axis."""
    if directions.shape[-1] != coefficients.shape[-1]:
        raise ValueError("directions and coefficients must have the same component count")
    return mean + xp.einsum("...c,vdc->...vd", coefficients, directions)


def pose_blend_shapes(
    rotations: Float[Array, "*batch J 3 3"],
    directions: Float[Array, "P V*3"],
    *,
    xp: Any,
) -> Float[Array, "*batch V 3"]:
    """Blend root-excluded joint rotation deviations into vertex offsets."""
    batch_shape = rotations.shape[:-3]
    identity = ops.eye_as(rotations, batch_dims=(*batch_shape, 1), xp=xp)
    features = (rotations[..., 1:, :, :] - identity).reshape(*batch_shape, -1)
    if features.shape[-1] != directions.shape[0]:
        raise ValueError("directions do not match the non-root joint rotations")
    return (features @ directions).reshape(*batch_shape, -1, 3)


__all__ = ["blend_shapes", "pose_blend_shapes"]
