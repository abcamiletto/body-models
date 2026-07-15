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
def test_runtime_zeros_have_independent_mutable_storage() -> None:
    numpy_zeros = NumpyRuntime().zeros((2, 3), like=np.zeros(()))
    numpy_zeros[0, 0] = 1
    np.testing.assert_array_equal(numpy_zeros, [[1, 0, 0], [0, 0, 0]])

    torch = pytest.importorskip("torch")
    torch_zeros = TorchRuntime().zeros((2, 3), like=torch.zeros(()))
    torch_zeros[0, 0] = 1
    torch.testing.assert_close(
        torch_zeros,
        torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch_zeros.dtype),
    )


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

    assert runtime.skinning_backend == "warp"
    assert runtime.xp.__name__ == "torch"


@pytest.mark.fast
@pytest.mark.parametrize("model_type", ["soma", "smpl"])
def test_soma_is_a_jax_pytree(model_type) -> None:
    jax = pytest.importorskip("jax")

    from body_models.soma.jax import SOMA

    model = SOMA(model_type=model_type)
    assert all(leaf is not model for leaf in jax.tree_util.tree_leaves(model))
    assert jax.jit(lambda value: value.num_vertices)(model) == model.num_vertices
