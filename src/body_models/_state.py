"""Materialize immutable model data for an array backend.

Backend-specific operations live on materialized state objects when they own
model-lifetime prepared data, as ``SparseLinear`` does. Stateless operations
whose inputs arrive per call are lowered by ``ArrayRuntime`` instead.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np

_JAX_DATACLASSES: set[type] = set()


def numpy_state(value: Any) -> Any:
    """Materialize model data for NumPy."""
    from body_models._common import sparse

    if isinstance(value, sparse.SparseMatrix):
        from body_models._common import sparse_numpy

        return sparse_numpy.SparseLinear(value)

    if is_dataclass(value):
        cls = type(value)
        return cls(**{field.name: numpy_state(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, list):
        return [numpy_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(numpy_state(item) for item in value)
    if isinstance(value, dict):
        return {key: numpy_state(item) for key, item in value.items()}
    return value


def torch_state(value: Any) -> Any:
    """Recursively register model arrays as PyTorch buffers."""
    import torch
    from torch import nn

    from body_models._common import sparse
    from body_models._torch_state import StateMapping, StateSequence

    if isinstance(value, sparse.SparseMatrix):
        from body_models._common import sparse_torch

        return sparse_torch.SparseLinear(value)

    if is_dataclass(value):
        module = nn.Module()
        for field in fields(value):
            setattr(module, field.name, torch_state(getattr(value, field.name)))
        return module

    if isinstance(value, dict):
        converted = {key: torch_state(item) for key, item in value.items()}
        if any(isinstance(item, torch.Tensor | nn.Module) for item in converted.values()):
            return StateMapping(converted)
        return converted

    if isinstance(value, list):
        converted = [torch_state(item) for item in value]
        if any(isinstance(item, torch.Tensor | nn.Module) for item in converted):
            return StateSequence(converted)
        return converted

    if isinstance(value, tuple):
        converted = tuple(torch_state(item) for item in value)
        if any(isinstance(item, torch.Tensor | nn.Module) for item in converted):
            return StateSequence(converted)
        return converted

    if isinstance(value, np.ndarray | torch.Tensor):
        return nn.Buffer(torch.as_tensor(value), persistent=True)

    return value


def jax_state(value: Any) -> Any:
    """Convert model arrays to JAX arrays while preserving dataclass types."""
    import jax
    import jax.numpy as jnp

    from body_models._common import sparse

    if isinstance(value, sparse.SparseMatrix):
        from body_models._common import sparse_jax

        materialized = sparse_jax.SparseLinear.from_matrix(value)
        _register_jax_dataclass(type(materialized), jax)
        return materialized

    if is_dataclass(value):
        cls = type(value)
        _register_jax_dataclass(cls, jax)
        return cls(**{field.name: jax_state(getattr(value, field.name)) for field in fields(value)})

    if isinstance(value, list):
        return [jax_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(jax_state(item) for item in value)
    if isinstance(value, dict):
        return {key: jax_state(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return jnp.asarray(value)
    return value


def register_jax_state(value: Any) -> None:
    """Restore pytree registrations for already-materialized JAX state."""
    import jax

    _register_jax_state(value, jax)


def _register_jax_state(value: Any, jax: Any) -> None:
    if is_dataclass(value):
        _register_jax_dataclass(type(value), jax)
        for field in fields(value):
            _register_jax_state(getattr(value, field.name), jax)
    elif isinstance(value, list | tuple):
        for item in value:
            _register_jax_state(item, jax)
    elif isinstance(value, dict):
        for item in value.values():
            _register_jax_state(item, jax)


def _register_jax_dataclass(cls: type, jax: Any) -> None:
    if cls in _JAX_DATACLASSES:
        return

    def flatten(obj):
        children = []
        child_names = []
        static = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            leaves = jax.tree_util.tree_leaves(value)
            if leaves and all(isinstance(leaf, jax.Array) for leaf in leaves):
                children.append(value)
                child_names.append(field.name)
            else:
                static[field.name] = value
        static_leaves, static_tree = jax.tree_util.tree_flatten(static)
        return tuple(children), (tuple(child_names), static_tree, tuple(static_leaves))

    def unflatten(aux_data, children):
        child_names, static_tree, static_leaves = aux_data
        values = jax.tree_util.tree_unflatten(static_tree, static_leaves)
        values.update(zip(child_names, children, strict=True))
        return cls(**values)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    _JAX_DATACLASSES.add(cls)


__all__ = ["jax_state", "numpy_state", "register_jax_state", "torch_state"]
