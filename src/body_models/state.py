"""Materialize immutable model data for an array backend."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Protocol

import numpy as np

_JAX_DATACLASSES: set[type] = set()


class StateMaterializer(Protocol):
    """Convert loaded model data into backend-managed state."""

    def __call__(self, value: Any) -> Any: ...


def numpy_state(value: Any) -> Any:
    """Keep loaded NumPy model data unchanged."""
    return value


def torch_state(value: Any) -> Any:
    """Recursively register model arrays as PyTorch buffers."""
    import torch
    import torch.nn as nn

    from body_models._torch_state import StateMapping, StateSequence

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
        return nn.Buffer(torch.as_tensor(value))

    return value


def jax_state(value: Any) -> Any:
    """Convert model arrays to JAX arrays while preserving dataclass types."""
    import jax
    import jax.numpy as jnp

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
        return tuple(children), (tuple(child_names), static)

    def unflatten(aux_data, children):
        child_names, static = aux_data
        values = dict(static)
        values.update(zip(child_names, children, strict=True))
        return cls(**values)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    _JAX_DATACLASSES.add(cls)


__all__ = ["StateMaterializer", "jax_state", "numpy_state", "torch_state"]
