"""Sparse linear algebra shared by learned corrective models."""

import numpy as np
import pytest

from body_models import _state as state
from body_models._common import sparse


def _weights() -> tuple[sparse.SparseMatrix, np.ndarray]:
    dense = np.array(
        [
            [1.0, 0.0, -2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0],
            [0.0, -6.0, 0.0],
        ],
        dtype=np.float32,
    )
    return sparse.from_dense(dense), dense


@pytest.mark.fast
def test_numpy_sparse_linear_matches_dense() -> None:
    weights, dense = _weights()
    inputs = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
    linear = state.numpy_state(weights)

    actual = sparse.linear(inputs, linear)

    np.testing.assert_array_equal(actual, inputs @ dense)
    exported = linear.to_coo()
    reconstructed = np.zeros(exported.shape, dtype=exported.values.dtype)
    reconstructed[exported.row_indices, exported.column_indices] = exported.values
    np.testing.assert_array_equal(reconstructed, dense)


@pytest.mark.fast
def test_torch_sparse_linear_matches_dense_under_compile() -> None:
    torch = pytest.importorskip("torch")
    weights, dense = _weights()
    linear = state.torch_state(weights).to(dtype=torch.float64)
    inputs = torch.arange(16, dtype=torch.float64).reshape(2, 2, 4).requires_grad_()

    actual = torch.compile(sparse.linear, backend="eager", fullgraph=True)(inputs, linear)
    expected = inputs @ torch.as_tensor(dense, dtype=torch.float64)

    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), inputs)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), inputs)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.slow
def test_torch_sparse_linear_compiles_with_inductor() -> None:
    torch = pytest.importorskip("torch")
    weights, dense = _weights()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    linear = state.torch_state(weights).to(device=device)
    inputs = torch.arange(16, dtype=torch.float32, device=device).reshape(2, 2, 4).requires_grad_()

    def loss(value):
        return sparse.linear(value, linear).square().sum()

    actual = torch.compile(loss, fullgraph=True)(inputs)
    actual_grad = torch.autograd.grad(actual, inputs)[0]
    expected = (inputs @ torch.as_tensor(dense, device=device)).square().sum()
    expected_grad = torch.autograd.grad(expected, inputs)[0]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.fast
def test_jax_sparse_linear_matches_dense_under_jit() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    weights, dense = _weights()
    linear = state.jax_state(weights)
    inputs = jnp.arange(16, dtype=jnp.float32).reshape(2, 2, 4)

    actual = jax.jit(sparse.linear)(inputs, linear)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(inputs @ dense))
