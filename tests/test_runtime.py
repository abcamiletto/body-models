"""Contracts shared by model runtimes."""

import pickle
import subprocess
import sys

import model_cases
import numpy as np
import pytest

from body_models._runtime import NumpyRuntime, TorchRuntime


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
def test_model_class_identity_is_backend_independent() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    from body_models import create_model
    from body_models.g1 import G1

    models = [G1(), create_model("g1", runtime="torch"), create_model("g1", runtime="jax")]

    assert all(type(model) is G1 for model in models)
    assert [model.runtime.backend for model in models] == ["numpy", "torch", "jax"]


@pytest.mark.fast
def test_model_pickle_uses_public_class_identity() -> None:
    from body_models.g1 import G1

    model = pickle.loads(pickle.dumps(G1()))

    assert type(model) is G1
    assert type(model).__module__ == "body_models.g1"


@pytest.mark.fast
def test_pickled_jax_model_jits_in_a_fresh_process() -> None:
    pytest.importorskip("jax")
    from body_models.g1 import G1

    model = G1(runtime="jax")
    program = """
import pickle
import sys

import jax

model = pickle.loads(sys.stdin.buffer.read())
print(jax.jit(lambda value: value.num_vertices)(model))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        input=pickle.dumps(model),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.decode().strip() == str(model.num_vertices)


@pytest.mark.fast
def test_registered_model_pytree_preserves_non_jax_runtime() -> None:
    jax = pytest.importorskip("jax")
    pytest.importorskip("torch")
    from body_models.g1 import G1

    model = G1(runtime=TorchRuntime("warp"))
    G1(runtime="jax")
    restored = jax.tree_util.tree_map(lambda value: value, model)

    assert type(restored) is G1
    assert restored.runtime == TorchRuntime("warp")
    assert restored._weights is model._weights


@pytest.mark.fast
def test_torch_module_manages_model_state() -> None:
    torch = pytest.importorskip("torch")
    from body_models.g1 import G1

    model = G1(runtime="torch")
    module = model.as_module()
    module.double()

    assert isinstance(module, torch.nn.Module)
    assert model.as_module() is module
    assert module.model is model
    assert module._weights is model._weights
    assert "_weights.vertices" in module.state_dict()
    assert model._weights.vertices.dtype == torch.float64

    restored_model = pickle.loads(pickle.dumps(model))
    restored_module = restored_model.as_module()
    assert restored_model.as_module() is restored_module
    assert restored_module.model is restored_model


@pytest.mark.fast
@pytest.mark.parametrize("model_type", ["soma", "smpl"])
def test_soma_is_a_jax_pytree(model_type) -> None:
    jax = pytest.importorskip("jax")

    from body_models.soma import SOMA

    model = SOMA(model_type=model_type, runtime="jax")
    assert all(leaf is not model for leaf in jax.tree_util.tree_leaves(model))
    assert jax.jit(lambda value: value.num_vertices)(model) == model.num_vertices


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_jax_model_pytree_round_trip(name, model_class, kwargs) -> None:
    jax = pytest.importorskip("jax")
    model = model_class(**kwargs, runtime="jax")

    leaves, tree = jax.tree_util.tree_flatten(model)
    restored = jax.tree_util.tree_unflatten(tree, leaves)

    assert type(restored) is type(model), name
    assert restored.runtime == model.runtime
    assert restored.num_vertices == model.num_vertices
    assert restored.joint_names == model.joint_names
    parameters = jax.jit(lambda value: value.get_rest_pose())(restored)
    assert parameters.keys() == model.parameter_spec.keys()
