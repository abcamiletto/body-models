"""Common utilities for multi-backend array operations."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np

Array = Any
_JAX_DATACLASSES = set()

__all__ = ["Array", "eye_as", "get_namespace", "jaxify", "set", "torchify", "zeros_as"]


def torchify(obj: Any) -> Any:
    """Convert dataclass arrays to torch buffers inside an nn.Module."""
    import torch
    import torch.nn as nn

    def convert_leaf(value):
        if isinstance(value, np.ndarray | torch.Tensor):
            return nn.Buffer(torch.as_tensor(value))
        return value

    def convert_dataclass(cls, values):
        module = nn.Module()
        for name, value in values.items():
            setattr(module, name, value)
        return module

    return _map_structure(obj, convert_leaf, convert_dataclass)


def jaxify(obj: Any) -> Any:
    """Convert dataclass arrays to JAX arrays, preserving dataclass types."""
    import jax
    import jax.numpy as jnp

    def convert_leaf(value):
        if isinstance(value, np.ndarray):
            return jnp.asarray(value)
        return value

    def convert_dataclass(cls, values):
        _register_jax_dataclass(cls, jax)
        return cls(**values)

    return _map_structure(obj, convert_leaf, convert_dataclass)


def _register_jax_dataclass(cls: type, jax: Any) -> None:
    if cls in _JAX_DATACLASSES:
        return

    def flatten(obj):
        children = []
        child_names = []
        static = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            if isinstance(value, jax.Array):
                children.append(value)
                child_names.append(field.name)
            else:
                static[field.name] = value
        return tuple(children), (tuple(child_names), static)

    def unflatten(aux_data, children):
        child_names, static = aux_data
        values = dict(static)
        values.update(zip(child_names, children, strict=True))
        return cls(**values)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    _JAX_DATACLASSES.add(cls)


def _map_structure(obj: Any, convert_leaf, convert_dataclass) -> Any:
    if is_dataclass(obj):
        values = {
            field.name: _map_structure(getattr(obj, field.name), convert_leaf, convert_dataclass)
            for field in fields(obj)
        }
        return convert_dataclass(type(obj), values)

    if isinstance(obj, list):
        return [_map_structure(item, convert_leaf, convert_dataclass) for item in obj]

    if isinstance(obj, tuple):
        return tuple(_map_structure(item, convert_leaf, convert_dataclass) for item in obj)

    if isinstance(obj, dict):
        return {key: _map_structure(value, convert_leaf, convert_dataclass) for key, value in obj.items()}

    return convert_leaf(obj)


def get_namespace(array: Array) -> Any:
    namespace = getattr(array, "__array_namespace__", None)
    if namespace is not None:
        return namespace()

    if type(array).__module__.startswith("torch"):
        import torch

        return torch

    raise TypeError(f"Unsupported array type '{type(array).__name__}'.")


def set(array: Array, slices: tuple, values: Array, *, copy: bool = True, xp: Any = None) -> Array:
    """Set elements of an array in a backend-independent way."""
    if xp is None:
        xp = get_namespace(array)

    if "jax" in xp.__name__:
        return array.at[slices].set(values)

    if copy:
        array = array.clone() if "torch" in xp.__name__ else xp.asarray(array, copy=True)

    array[slices] = values
    return array


def zeros_as(ref: Array, *, shape: tuple[int, ...], xp: Any = None) -> Array:
    """Create a zero array with ref's backend/device/dtype and a target shape."""
    if xp is None:
        xp = get_namespace(ref)
    z = xp.zeros_like(ref)
    base = z if z.ndim == 0 else xp.reshape(z, (-1,))[:1]
    return xp.broadcast_to(base, shape)


def eye_as(ref: Array, *, batch_dims: tuple[int, ...], xp: Any = None) -> Array:
    """Create batched identity matrices using ref's backend/device/dtype."""
    if xp is None:
        xp = get_namespace(ref)
    n = ref.shape[-1]
    eye = zeros_as(ref, shape=(*batch_dims, n, n), xp=xp)
    one = 1 if "torch" in xp.__name__ else 1.0
    for i in range(n):
        eye = set(eye, (..., i, i), one, xp=xp)
    return eye
