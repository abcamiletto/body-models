import numpy as np
import pytest

import model_cases


@pytest.mark.parametrize(
    ("name", "numpy_model", "torch_model", "jax_model", "model_path", "kwargs"), model_cases.MODELS
)
def test_torch_and_jax_match_numpy(name, numpy_model, torch_model, jax_model, model_path, kwargs) -> None:
    if not model_path.exists():
        pytest.skip(f"Missing model asset: {model_path}")

    numpy_instance = numpy_model(model_path=model_path, **kwargs)
    numpy_params = numpy_instance.get_rest_pose(batch_size=2, dtype=np.float32)
    expected = numpy_instance.forward_vertices(**numpy_params)

    torch = pytest.importorskip("torch")
    torch_instance = torch_model(model_path=model_path, **kwargs)
    torch_params = torch_instance.get_rest_pose(batch_size=2, dtype=torch.float32)
    with torch.no_grad():
        torch_vertices = torch_instance.forward_vertices(**torch_params)
    np.testing.assert_allclose(torch_vertices.numpy(), expected, rtol=1e-4, atol=1e-4)

    pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp

    jax_instance = jax_model(model_path=model_path, **kwargs)
    jax_params = jax_instance.get_rest_pose(batch_size=2, dtype=jnp.float32)
    jax_vertices = jax_instance.forward_vertices(**jax_params)
    np.testing.assert_allclose(np.asarray(jax_vertices), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("name", "numpy_model", "torch_model", "_jax_model", "model_path", "kwargs"), model_cases.MODELS
)
def test_kernels_match_default(name, numpy_model, torch_model, _jax_model, model_path, kwargs) -> None:
    if not model_path.exists():
        pytest.skip(f"Missing model asset: {model_path}")

    numpy_instance = numpy_model(model_path=model_path, **kwargs)
    for kernel in getattr(numpy_instance, "kernels", ())[1:]:
        params = numpy_instance.get_rest_pose(batch_size=2, dtype=np.float32)
        expected = numpy_instance.forward_vertices(**params)
        actual = numpy_model(model_path=model_path, kernel=kernel, **kwargs).forward_vertices(**params)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    torch = pytest.importorskip("torch")
    torch_instance = torch_model(model_path=model_path, **kwargs)
    for kernel in getattr(torch_instance, "kernels", ())[1:]:
        params = torch_instance.get_rest_pose(batch_size=2, dtype=torch.float32)
        with torch.no_grad():
            expected = torch_instance.forward_vertices(**params)
            actual = torch_model(model_path=model_path, kernel=kernel, **kwargs).forward_vertices(**params)
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-4)
