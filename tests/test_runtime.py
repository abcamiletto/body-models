"""Contracts shared by model runtimes."""

import pickle

import numpy as np
import pytest

from body_models.runtime import JaxRuntime, NumpyRuntime, TorchRuntime


@pytest.mark.fast
def test_runtimes_expand_compact_skinning_weights() -> None:
    indices = np.array([[0, 2, -1], [1, 1, 2]], dtype=np.int64)
    weights = np.array([[0.25, 0.75, 0.0], [0.2, 0.3, 0.5]], dtype=np.float32)
    expected = np.array([[0.25, 0.0, 0.75], [0.0, 0.5, 0.5]], dtype=np.float32)

    numpy = NumpyRuntime()
    actual = numpy.expand_skinning_weights(indices, weights, num_joints=3)
    np.testing.assert_array_equal(actual, expected)

    torch = pytest.importorskip("torch")
    torch_runtime = TorchRuntime()
    torch_like = torch.zeros((), dtype=torch.float32)
    torch_indices = torch_runtime.asarray(indices, like=torch_like, dtype=torch.int64)
    torch_weights = torch_runtime.asarray(weights, like=torch_like)
    actual = torch_runtime.expand_skinning_weights(torch_indices, torch_weights, num_joints=3)
    np.testing.assert_array_equal(actual.numpy(), expected)

    pytest.importorskip("jax")
    import jax.numpy as jnp

    jax_runtime = JaxRuntime()
    jax_like = jnp.zeros((), dtype=jnp.float32)
    jax_indices = jax_runtime.asarray(indices, like=jax_like, dtype=jnp.int32)
    jax_weights = jax_runtime.asarray(weights, like=jax_like)
    actual = jax_runtime.expand_skinning_weights(jax_indices, jax_weights, num_joints=3)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.fast
def test_runtime_array_creation_follows_reference_dtype() -> None:
    numpy = NumpyRuntime()
    reference = np.zeros((), dtype=np.float64)
    assert numpy.asarray([1.0], like=reference).dtype == np.float64
    assert numpy.zeros((2, 3), like=reference).dtype == np.float64

    torch = pytest.importorskip("torch")
    torch_runtime = TorchRuntime()
    reference = torch.zeros((), dtype=torch.float64)
    assert torch_runtime.asarray([1.0], like=reference).dtype == torch.float64
    assert torch_runtime.zeros((2, 3), like=reference).dtype == torch.float64


@pytest.mark.fast
def test_compact_skinning_ignores_padding_slots() -> None:
    runtime = NumpyRuntime()
    vertices = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    transforms = np.broadcast_to(np.eye(4, dtype=np.float32), (1, 2, 4, 4)).copy()
    transforms[0, 1, :3, 3] = 100.0
    indices = np.array([[0, -1]], dtype=np.int64)
    weights = np.array([[1.0, 7.0]], dtype=np.float32)

    actual = runtime.compact_linear_blend_skinning(
        vertices,
        transforms,
        joint_indices=indices,
        joint_weights=weights,
    )

    np.testing.assert_array_equal(actual, vertices[None])


@pytest.mark.fast
def test_runtime_is_serializable() -> None:
    runtime = pickle.loads(pickle.dumps(TorchRuntime("warp")))

    assert runtime.kernel == "warp"
    assert runtime.xp.__name__ == "torch"
