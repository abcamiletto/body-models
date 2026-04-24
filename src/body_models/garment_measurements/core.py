"""Backend-agnostic GarmentMeasurements PCA body model computation."""

from __future__ import annotations

from typing import Any

from array_api_compat import get_namespace
from jaxtyping import Float
from nanomanifold import SE3, SO3

from ..rotations import RotationType

Array = Any


def forward_vertices(
    mean_vertices: Float[Array, "V 3"],
    components: Float[Array, "V 3 C"],
    eigenvalues: Float[Array, "C"],
    shape: Float[Array, "B C"],
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Evaluate PCA body vertices [B, V, 3].

    The upstream shape sampler interprets public PCA controls as standard
    deviation units and multiplies them by sqrt(eigenvalue) before evaluating
    the PCA matrix.
    """
    assert shape.ndim == 2
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(shape)

    if vertex_indices is not None:
        vertex_indices = xp.asarray(vertex_indices)
        mean_vertices = mean_vertices[vertex_indices]
        components = components[vertex_indices]

    scaled_shape = shape * xp.sqrt(eigenvalues)[None]
    vertices = mean_vertices[None] + xp.einsum("bc,vdc->bvd", scaled_shape, components)

    if global_rotation is None and global_translation is None:
        return vertices

    quat, translation = _global_quat_translation(
        xp=xp,
        ref=vertices,
        batch_size=vertices.shape[0],
        global_rotation=global_rotation,
        global_translation=global_translation,
        rotation_type=rotation_type,
    )
    return SE3.transform_points(SE3.from_rt(quat, translation, xp=xp), vertices, xp=xp)


def _global_quat_translation(
    *,
    xp: Any,
    ref: Float[Array, "..."],
    batch_size: int,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None,
    global_translation: Float[Array, "B 3"] | None,
    rotation_type: RotationType,
) -> tuple[Float[Array, "B 4"], Float[Array, "B 3"]]:
    if global_rotation is None:
        quat = SO3.identity_as(ref, batch_dims=(batch_size,), rotation_type="quat", xp=xp)
    else:
        global_rotation = _match_dtype(global_rotation, ref, xp=xp)
        quat = SO3.convert(global_rotation, src=rotation_type, dst="quat", xp=xp)

    if global_translation is None:
        zero = xp.zeros_like(ref.reshape(-1)[:1])
        translation = xp.broadcast_to(zero, (batch_size, 3))
    else:
        translation = _match_dtype(global_translation, ref, xp=xp)
    return quat, translation


def _match_dtype(value: Array, ref: Array, *, xp: Any) -> Array:
    zero = xp.zeros_like(ref.reshape(-1)[:1])
    return value + zero
