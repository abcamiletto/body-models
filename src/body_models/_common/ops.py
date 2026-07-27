"""Common utilities for multi-backend array operations."""

from __future__ import annotations

from typing import Any

import array_api_compat
from jaxtyping import Float, Int, Num

Array = Any
__all__ = ["Array", "at_set", "eye_as", "take_along_axis", "zeros_as"]


def at_set(
    array: Num[Array, "..."],
    slices: tuple,
    values: Num[Array, "..."] | float,
    *,
    copy: bool = True,
    xp: Any,
) -> Num[Array, "..."]:
    """Set elements of an array in a backend-independent way."""
    if array_api_compat.is_jax_array(array):
        return array.at[slices].set(values)

    if copy:
        array = array.clone() if array_api_compat.is_torch_array(array) else xp.asarray(array, copy=True)

    array[slices] = values
    return array


def take_along_axis(
    array: Num[Array, "..."],
    indices: Int[Array, "..."],
    axis: int,
    *,
    xp: Any,
) -> Num[Array, "..."]:
    """Select values along one axis using backend-native naming."""
    if array_api_compat.is_torch_array(array):
        return xp.take_along_dim(array, indices, dim=axis)
    return xp.take_along_axis(array, indices, axis=axis)


def zeros_as(
    ref: Num[Array, "..."],
    *,
    shape: tuple[int, ...],
    dtype: Any | None = None,
    xp: Any,
) -> Num[Array, "..."]:
    """Create a zero array with ref's backend/device/dtype and a target shape."""
    if dtype is None:
        dtype = ref.dtype

    device = getattr(ref, "device", None)
    return xp.zeros(shape, dtype=dtype) if device is None else xp.zeros(shape, dtype=dtype, device=device)


def eye_as(
    ref: Float[Array, "... N"],
    *,
    batch_dims: tuple[int, ...],
    xp: Any,
) -> Float[Array, "*batch N N"]:
    """Create batched identity matrices using ref's backend/device/dtype."""
    n = ref.shape[-1]
    eye = zeros_as(ref, shape=(*batch_dims, n, n), xp=xp)
    for i in range(n):
        eye = at_set(eye, (..., i, i), 1.0, xp=xp)
    return eye
