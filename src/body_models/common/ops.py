"""Common utilities for multi-backend array operations."""

from __future__ import annotations

from typing import Any

from jaxtyping import Float, Num

Array = Any
__all__ = ["Array", "eye_as", "get_namespace", "set", "zeros_as"]


def get_namespace(array: Num[Array, "..."]) -> Any:
    namespace = getattr(array, "__array_namespace__", None)
    if namespace is not None:
        return namespace()

    if type(array).__module__.startswith("torch"):
        import torch

        return torch

    raise TypeError(f"Unsupported array type '{type(array).__name__}'.")


def set(
    array: Num[Array, "..."],
    slices: tuple,
    values: Num[Array, "..."] | float | int,
    *,
    copy: bool = True,
    xp: Any = None,
) -> Num[Array, "..."]:
    """Set elements of an array in a backend-independent way."""
    if xp is None:
        xp = get_namespace(array)

    if "jax" in xp.__name__:
        return array.at[slices].set(values)

    if copy:
        array = array.clone() if "torch" in xp.__name__ else xp.asarray(array, copy=True)

    array[slices] = values
    return array


def zeros_as(
    ref: Num[Array, "..."],
    *,
    shape: tuple[int, ...],
    dtype: Any | None = None,
    xp: Any = None,
) -> Num[Array, "..."]:
    """Create a zero array with ref's backend/device/dtype and a target shape."""
    if xp is None:
        xp = get_namespace(ref)
    if dtype is None:
        dtype = ref.dtype

    if "torch" in xp.__name__:
        return xp.zeros(shape, dtype=dtype, device=ref.device)

    zeros = xp.zeros(shape, dtype=dtype)
    if "jax" not in xp.__name__:
        return zeros

    device = getattr(ref, "device", None)
    if device is None:
        return zeros

    import jax

    return jax.device_put(zeros, device)


def eye_as(
    ref: Float[Array, "... N"],
    *,
    batch_dims: tuple[int, ...],
    xp: Any = None,
) -> Float[Array, "*batch N N"]:
    """Create batched identity matrices using ref's backend/device/dtype."""
    if xp is None:
        xp = get_namespace(ref)
    n = ref.shape[-1]
    eye = zeros_as(ref, shape=(*batch_dims, n, n), xp=xp)
    one = 1 if "torch" in xp.__name__ else 1.0
    for i in range(n):
        eye = set(eye, (..., i, i), one, xp=xp)
    return eye
