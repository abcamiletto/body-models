"""Backend model-state materialization contracts."""

from dataclasses import dataclass

import numpy as np
import pytest

from body_models._state import numpy_state, torch_state


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


@pytest.mark.fast
def test_materialized_arrays_have_independent_storage() -> None:
    source = np.ones(2, dtype=np.float32)
    numpy = numpy_state(source)
    numpy[0] = 2
    assert source[0] == 1

    pytest.importorskip("torch")
    torch_array = torch_state(source)
    torch_array[0] = 3
    assert source[0] == 1


@pytest.mark.fast
def test_warp_plan_is_rebuilt_after_loading_state() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    from body_models._common import skinning, warp

    weights = torch.ones((2, 2), dtype=torch.float32)
    first = warp.prepare_compact_skinning(skinning.CompactSkinning(torch.tensor([[0, 1], [0, 1]]), weights))
    second = warp.prepare_compact_skinning(skinning.CompactSkinning(torch.tensor([[1, 0], [1, 0]]), weights))

    first.load_state_dict(second.state_dict())

    torch.testing.assert_close(first._plan_permutation, second._plan_permutation)
    torch.testing.assert_close(first._plan_chunk_joints, second._plan_chunk_joints)
