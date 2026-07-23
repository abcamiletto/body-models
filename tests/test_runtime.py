"""Contracts shared by model runtimes."""

import pickle

import numpy as np
import pytest

import model_cases
from body_models.runtime import NumpyRuntime, TorchRuntime


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


@pytest.mark.parametrize(("name", "_numpy", "_torch", "jax_model", "kwargs"), model_cases.MODELS)
def test_jax_model_pytree_round_trip(name, _numpy, _torch, jax_model, kwargs) -> None:
    jax = pytest.importorskip("jax")
    model = jax_model(**kwargs)

    leaves, tree = jax.tree_util.tree_flatten(model)
    restored = jax.tree_util.tree_unflatten(tree, leaves)

    assert type(restored) is type(model), name
    assert restored.num_vertices == model.num_vertices
    assert restored.joint_names == model.joint_names
    assert restored.get_rest_pose().keys() == model.get_rest_pose().keys()
