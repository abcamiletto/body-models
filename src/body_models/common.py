"""Common utilities for multi-backend array operations."""

from typing import Any

import numpy as np
from array_api_compat import get_namespace
from jaxtyping import Float, Int

Array = Any


def simplify_mesh(
    vertices: Float[np.ndarray, "V 3"],
    faces: Int[np.ndarray, "F 3"],
    target_faces: int,
) -> tuple[Float[np.ndarray, "V2 3"], Int[np.ndarray, "F2 3"], Int[np.ndarray, "V2"]]:
    """Simplify mesh using quadric decimation.

    Args:
        vertices: [V, 3] vertex positions
        faces: [F, 3] face indices
        target_faces: target number of faces

    Returns:
        new_vertices: [V', 3] simplified vertex positions
        new_faces: [F', 3] simplified face indices
        vertex_map: [V'] index of nearest original vertex for each new vertex
    """
    import pyfqmr
    from scipy.spatial import KDTree

    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(vertices, faces)
    simplifier.simplify_mesh(target_count=target_faces, aggressiveness=7, preserve_border=True)
    new_vertices, new_faces, _ = simplifier.getMesh()

    new_vertices = np.asarray(new_vertices, dtype=np.float32)
    new_faces = np.asarray(new_faces, dtype=np.int32)

    tree = KDTree(vertices)
    _, vertex_map = tree.query(new_vertices)
    vertex_map = np.asarray(vertex_map, dtype=np.int64)

    return new_vertices, new_faces, vertex_map


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
