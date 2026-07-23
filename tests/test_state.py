"""Backend model-state materialization contracts."""

from dataclasses import dataclass

import numpy as np
import pytest

from body_models.state import torch_state


@dataclass(frozen=True)
class _Leaf:
    values: np.ndarray


@dataclass(frozen=True)
class _Tree:
    leaves: dict[str, _Leaf]
    arrays: dict[str, np.ndarray]


@pytest.mark.fast
def test_torch_state_registers_nested_arrays() -> None:
    torch = pytest.importorskip("torch")
    state = torch_state(
        _Tree(
            leaves={"low": _Leaf(np.ones(2, dtype=np.float32))},
            arrays={"indices": np.arange(2)},
        )
    )

    assert list(state.state_dict()) == ["leaves.low.values", "arrays.indices"]
    state.to(dtype=torch.float64)
    assert state.leaves["low"].values.dtype == torch.float64
    assert state.arrays["indices"].device == state.leaves["low"].values.device
