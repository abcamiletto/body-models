"""Model runtime behavior."""

import pickle
import subprocess
import sys

import model_cases
import numpy as np
import pytest

from body_models._common import skinning
from body_models._runtime import JaxRuntime, NumpyRuntime, TorchRuntime


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
@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_runtime_rejects_unknown_state(backend) -> None:
    if backend != "numpy":
        pytest.importorskip(backend)

    runtime = {"numpy": NumpyRuntime, "torch": TorchRuntime, "jax": JaxRuntime}[backend]()

    with pytest.raises(TypeError, match="Unsupported model state leaf"):
        runtime._materialize(object())


@pytest.mark.fast
def test_jax_materialization_preserves_jax_arrays() -> None:
    jax = pytest.importorskip("jax")
    value = jax.numpy.ones(2)

    assert JaxRuntime()._materialize(value) is value


@pytest.mark.fast
def test_runtime_stop_gradient() -> None:
    numpy_value = np.ones(2, dtype=np.float32)
    assert NumpyRuntime().stop_gradient(numpy_value) is numpy_value

    torch = pytest.importorskip("torch")
    torch_value = torch.ones(2, requires_grad=True)
    assert not TorchRuntime().stop_gradient(torch_value).requires_grad

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    gradient = jax.grad(lambda value: JaxRuntime().stop_gradient(value).sum())(jnp.ones(2))
    np.testing.assert_array_equal(gradient, np.zeros(2, dtype=np.float32))


@pytest.mark.fast
def test_compact_skinning_ignores_padding_slots() -> None:
    runtime = NumpyRuntime()
    vertices = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    transforms = np.broadcast_to(np.eye(4, dtype=np.float32), (1, 2, 4, 4)).copy()
    transforms[0, 1, :3, 3] = 100.0
    indices = np.array([[0, -1]], dtype=np.int64)
    weights = np.array([[1.0, 7.0]], dtype=np.float32)

    actual = runtime._skin_vertices(
        vertices,
        transforms,
        skinning=skinning.CompactSkinning(indices, weights),
    )

    np.testing.assert_array_equal(actual, vertices[None])


@pytest.mark.fast
def test_runtime_is_serializable() -> None:
    runtime = pickle.loads(pickle.dumps(TorchRuntime(kernel_backend="warp")))

    assert runtime.kernel_backend == "warp"
    assert runtime.xp.__name__ == "torch"


def test_factory_returns_backend_specific_model() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("jax")
    from body_models import create_model
    from body_models.smpl.jax import SMPL as JaxSMPL
    from body_models.smpl.numpy import SMPL as NumpySMPL
    from body_models.smpl.torch import SMPL as TorchSMPL

    models = [
        create_model("smpl", gender="neutral"),
        create_model("smpl", runtime="torch", gender="neutral"),
        create_model("smpl", runtime="jax", gender="neutral"),
    ]

    assert [type(model) for model in models] == [NumpySMPL, TorchSMPL, JaxSMPL]
    assert [model.runtime.name for model in models] == ["numpy", "torch", "jax"]
    assert isinstance(models[1], torch.nn.Module)


def test_model_pickle_uses_public_class_identity() -> None:
    from body_models.smpl.numpy import SMPL

    model = pickle.loads(pickle.dumps(SMPL(gender="neutral")))

    assert type(model) is SMPL
    assert type(model).__module__ == "body_models.smpl.numpy"


def test_pickled_jax_model_jits_in_a_fresh_process() -> None:
    pytest.importorskip("jax")
    from body_models.smpl.jax import SMPL

    model = SMPL(gender="neutral")
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


def test_torch_model_manages_module_state() -> None:
    torch = pytest.importorskip("torch")
    from body_models.smpl.torch import SMPL

    model = SMPL(gender="neutral")
    model.double()

    assert isinstance(model, torch.nn.Module)
    assert "_weights.v_template" in model.state_dict()
    assert model._weights.v_template.dtype == torch.float64

    restored = pickle.loads(pickle.dumps(model))
    assert isinstance(restored, SMPL)
    assert "_weights.v_template" in restored.state_dict()


@pytest.mark.parametrize("model_type", ["soma", "smpl"])
def test_soma_is_a_jax_pytree(model_type) -> None:
    jax = pytest.importorskip("jax")

    from body_models.soma.jax import SOMA

    model = SOMA(model_type=model_type)
    assert all(leaf is not model for leaf in jax.tree_util.tree_leaves(model))
    assert jax.jit(lambda value: value.num_vertices)(model) == model.num_vertices


def test_soma_torch_model_owns_external_identity_model() -> None:
    torch = pytest.importorskip("torch")
    from body_models.smpl.torch import SMPL
    from body_models.soma.torch import SOMA

    model = SOMA(model_type="smpl")
    identity_model = model._identity_model

    assert isinstance(identity_model, SMPL)
    assert any(name.startswith("_identity_model._weights.") for name in model.state_dict())

    model.double()
    assert identity_model.rest_vertices.dtype == torch.float64

    restored = pickle.loads(pickle.dumps(model))
    assert isinstance(restored._identity_model, SMPL)
    assert any(name.startswith("_identity_model._weights.") for name in restored.state_dict())


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_jax_model_pytree_round_trip(name, model_class, kwargs) -> None:
    jax = pytest.importorskip("jax")
    jax_class = model_cases.backend_model_class(name, "jax")
    model = jax_class(**kwargs)

    leaves, tree = jax.tree_util.tree_flatten(model)
    restored = jax.tree_util.tree_unflatten(tree, leaves)

    assert type(restored) is type(model), name
    assert restored.runtime == model.runtime
    assert restored.num_vertices == model.num_vertices
    assert restored.joint_names == model.joint_names
    parameters = jax.jit(lambda value: value.get_rest_pose())(restored)
    assert parameters.keys() == model.parameter_spec.keys()
