"""PyTorch containers for recursively registered model state."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch import nn


class StateMapping(nn.Module, Mapping[str, Any]):
    """Mapping whose array values participate in the module lifecycle."""

    __hash__ = object.__hash__

    def __init__(self, values: Mapping[str, Any]) -> None:
        super().__init__()
        if any(not isinstance(key, str) for key in values):
            raise TypeError("Model state mappings must use string keys.")
        self._indices = {key: index for index, key in enumerate(sorted(values))}
        self._values = StateSequence([values[key] for key in self._indices])

    def __getitem__(self, key: str) -> Any:
        return self._values[self._indices[key]]

    def __iter__(self) -> Iterator[str]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


class StateSequence(nn.Module, Sequence[Any]):
    """Sequence whose array values participate in the module lifecycle."""

    def __init__(self, values: Sequence[Any]) -> None:
        super().__init__()
        self._length = len(values)
        self._static = {}
        for index, value in enumerate(values):
            name = str(index)
            if isinstance(value, torch.Tensor):
                self.register_buffer(name, value, persistent=True)
            elif isinstance(value, nn.Module):
                self.add_module(name, value)
            else:
                self._static[name] = value

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._length))]
        if not -self._length <= index < self._length:
            raise IndexError(index)
        index %= self._length
        name = str(index)
        return self._static[name] if name in self._static else getattr(self, name)

    def __len__(self) -> int:
        return self._length


__all__ = ["StateMapping", "StateSequence"]
