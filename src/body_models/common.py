"""Common utilities for multi-backend array operations."""

from typing import Any

from array_api_compat import get_namespace

Array = Any


def set(array: Array, slices: tuple, values: Array, *, copy: bool = True) -> Array:
    """Set elements of an array in a backend-independent way.

    Args:
        array: The array to modify.
        slices: Index/slice tuple. Use `np.index_exp` for readable syntax.
        values: Values to set at the specified indices.
        copy: If True (default), copy the array before modifying (NumPy/PyTorch only).
              JAX always returns a new array regardless of this flag.

    Returns:
        The modified array (new array for JAX, possibly same array for NumPy/PyTorch if copy=False).

    Examples:
        >>> import numpy as np
        >>> from body_models import common
        >>> arr = common.set(arr, np.index_exp[..., :3, :3], rotation)  # arr[..., :3, :3] = R
        >>> arr = common.set(arr, np.index_exp[:, 0], first_row)        # arr[:, 0] = first_row
    """
    # JAX arrays use functional update API via .at attribute
    if "jax" in type(array).__module__:
        return array.at[slices].set(values)

    # NumPy/PyTorch: use classic item assignment
    if copy:
        xp = get_namespace(array)
        # PyTorch: clone() preserves gradients, asarray() does not
        if "torch" in xp.__name__:
            array = array.clone()
        else:
            array = xp.asarray(array, copy=True)

    array[slices] = values
    return array
