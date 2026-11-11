"""Materialize immutable model data for an array backend.

Reusable backend-specific data lives on materialized state objects. Operation
execution is lowered by ``ArrayRuntime`` instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from body_models._common import kinematics, sparse

if TYPE_CHECKING:
    from body_models._runtime import KernelBackend

_JAX_DATACLASSES: set[type] = set()
_STATIC_LEAF_TYPES = (type(None), bool, int, float, str, bytes, np.generic, kinematics.KinematicTree)


def _rebuild(
    value: Any,
    *,
    array: Callable[[Any], Any],
    sparse_matrix: Callable[[sparse.SparseMatrix], Any],
    dataclass: Callable[[Any, dict[str, Any]], Any],
    container: Callable[[Any], Any] = lambda converted: converted,
) -> Any:
    """Rebuild a state tree bottom-up, converting each node with a backend hook.

    Static leaves are kept as-is. Dataclasses and containers are rebuilt from their
    converted children; anything else is an array leaf.
    """

    def walk(value: Any) -> Any:
        if isinstance(value, _STATIC_LEAF_TYPES):
            return value
        if isinstance(value, sparse.SparseMatrix):
            return sparse_matrix(value)
        if is_dataclass(value):
            return dataclass(value, {field.name: walk(getattr(value, field.name)) for field in fields(value)})
        if isinstance(value, dict):
            return container({key: walk(item) for key, item in value.items()})
        if isinstance(value, list | tuple):
            return container(type(value)(walk(item) for item in value))
        return array(value)

    return walk(value)


def numpy_state(value: Any) -> Any:
    """Materialize model data for NumPy."""
    from body_models._common import sparse_numpy

    def array(value: Any) -> np.ndarray:
        return value.copy() if isinstance(value, np.ndarray) else _unsupported_leaf(value)

    return _rebuild(
        value,
        array=array,
        sparse_matrix=sparse_numpy.SparseLinear,
        dataclass=lambda value, children: type(value)(**children),
    )


def torch_state(value: Any, *, kernel_backend: KernelBackend = "torch") -> Any:
    """Recursively register model arrays as PyTorch buffers."""
    import torch
    from torch import nn

    from body_models._common import skinning, sparse_torch
    from body_models._torch_state import StateMapping, StateSequence

    def array(value: Any) -> nn.Buffer:
        if isinstance(value, np.ndarray):
            value = torch.tensor(value)
        return nn.Buffer(value, persistent=True) if isinstance(value, torch.Tensor) else _unsupported_leaf(value)

    def dataclass(value: Any, children: dict[str, Any]) -> Any:
        module = nn.Module()
        for name, child in children.items():
            setattr(module, name, child)
        if isinstance(value, skinning.CompactSkinning) and kernel_backend == "warp":
            try:
                from body_models._common import warp
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError("Install body-models[warp] to use kernel_backend='warp'.") from exc
            return warp.prepare_compact_skinning(module)
        return module

    def container(converted: Any) -> Any:
        items = converted.values() if isinstance(converted, dict) else converted
        if not any(isinstance(item, torch.Tensor | nn.Module) for item in items):
            return converted
        return StateMapping(converted) if isinstance(converted, dict) else StateSequence(converted)

    return _rebuild(
        value, array=array, sparse_matrix=sparse_torch.SparseLinear, dataclass=dataclass, container=container
    )


def jax_state(value: Any) -> Any:
    """Convert model arrays to JAX arrays while preserving dataclass types."""
    import jax
    import jax.numpy as jnp

    from body_models._common import sparse_jax

    def array(value: Any) -> jax.Array:
        if isinstance(value, np.ndarray):
            return jnp.asarray(value)
        return value if isinstance(value, jax.Array) else _unsupported_leaf(value)

    def sparse_matrix(value: sparse.SparseMatrix) -> sparse_jax.SparseLinear:
        _register_jax_dataclass(sparse_jax.SparseLinear, jax)
        return sparse_jax.SparseLinear.from_matrix(value)

    def dataclass(value: Any, children: dict[str, Any]) -> Any:
        _register_jax_dataclass(type(value), jax)
        return type(value)(**children)

    return _rebuild(value, array=array, sparse_matrix=sparse_matrix, dataclass=dataclass)


def register_jax_state(value: Any) -> None:
    """Restore pytree registrations for already-materialized JAX state."""
    import jax

    def dataclass(value: Any, children: dict[str, Any]) -> Any:
        _register_jax_dataclass(type(value), jax)
        return value

    _rebuild(value, array=lambda value: value, sparse_matrix=lambda value: value, dataclass=dataclass)


def _unsupported_leaf(value: Any) -> Any:
    value_type = type(value)
    raise TypeError(f"Unsupported model state leaf: {value_type.__module__}.{value_type.__qualname__}")


def _register_jax_dataclass(cls: type, jax: Any) -> None:
    if cls in _JAX_DATACLASSES:
        return

    field_names = tuple(field.name for field in fields(cls))

    def flatten(obj):
        values = tuple(getattr(obj, name) for name in field_names)
        leaves, tree = jax.tree_util.tree_flatten(values)
        static = tuple((index, leaf) for index, leaf in enumerate(leaves) if isinstance(leaf, _STATIC_LEAF_TYPES))
        children = tuple(leaf for leaf in leaves if not isinstance(leaf, _STATIC_LEAF_TYPES))
        return children, (tree, static)

    def unflatten(aux_data, children):
        tree, static = aux_data
        leaves = list(children)
        for index, value in static:
            leaves.insert(index, value)
        values = jax.tree_util.tree_unflatten(tree, leaves)
        return cls(**dict(zip(field_names, values, strict=True)))

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    _JAX_DATACLASSES.add(cls)


__all__ = ["jax_state", "numpy_state", "register_jax_state", "torch_state"]
