"""Common utilities for multi-backend array operations."""

from __future__ import annotations

from typing import Any

from jaxtyping import Float, Num

Array = Any
__all__ = ["Array", "eye_as", "set", "zeros_as"]


def set(
    array: Num[Array, "..."],
    slices: tuple,
    values: Num[Array, "..."] | float | int,
    *,
    copy: bool = True,
    xp: Any,
) -> Num[Array, "..."]:
    """Set elements of an array in a backend-independent way."""
    if hasattr(array, "at"):
        return array.at[slices].set(values)

    if copy:
        array = array.clone() if hasattr(array, "clone") else xp.asarray(array, copy=True)

    array[slices] = values
    return array


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
        eye = set(eye, (..., i, i), 1.0, xp=xp)
    return eye
