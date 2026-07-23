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


class StateMapping(nn.Module, Mapping[str, Any]):
    """Mapping whose array values participate in the module lifecycle."""

    __hash__ = object.__hash__

    def __init__(self, values: Mapping[str, Any]) -> None:
        super().__init__()
        self._keys = tuple(values)
        self._static = {}
        for key, value in values.items():
            if not isinstance(key, str):
                raise TypeError("Model state mappings must use string keys.")
            _store(self, self._static, key, value)

    def __getitem__(self, key: str) -> Any:
        try:
            return self._static[key] if key in self._static else getattr(self, key)
        except KeyError as exc:
            raise KeyError(key) from exc
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __iter__(self) -> Iterator[str]:
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
