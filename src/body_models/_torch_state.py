"""PyTorch containers for recursively registered model state."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn


def _store(module: nn.Module, static: dict[str, Any], name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        module.register_buffer(name, value)
    elif isinstance(value, nn.Module):
        module.add_module(name, value)
    else:
        static[name] = value


class StateMapping(nn.Module, Mapping[Any, Any]):
    """Mapping whose array values participate in the module lifecycle."""

    __hash__ = object.__hash__

    def __init__(self, values: Mapping[Any, Any]) -> None:
        super().__init__()
        self._keys = tuple(values)
        self._indices = {key: index for index, key in enumerate(self._keys)}
        self._static = {}
        for index, value in enumerate(values.values()):
            _store(self, self._static, str(index), value)

    def __getitem__(self, key: Any) -> Any:
        try:
            index = self._indices[key]
        except KeyError as exc:
            raise KeyError(key) from exc
        name = str(index)
        return self._static[name] if name in self._static else getattr(self, name)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


class StateSequence(nn.Module):
    """Sequence whose array values participate in the module lifecycle."""

    def __init__(self, values: Sequence[Any]) -> None:
        super().__init__()
        self._length = len(values)
        self._static = {}
        for index, value in enumerate(values):
            _store(self, self._static, str(index), value)

    def __getitem__(self, index: int) -> Any:
        if not -self._length <= index < self._length:
            raise IndexError(index)
        index %= self._length
        name = str(index)
        return self._static[name] if name in self._static else getattr(self, name)

    def __len__(self) -> int:
        return self._length


__all__ = ["StateMapping", "StateSequence"]
