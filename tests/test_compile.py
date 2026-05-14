import numpy as np
import pytest

import model_cases


@pytest.mark.parametrize(
    ("name", "_numpy_model", "torch_model", "jax_model", "model_path", "kwargs"), model_cases.MODELS
)
def test_torch_compile_and_jax_jit(name, _numpy_model, torch_model, jax_model, model_path, kwargs) -> None:
    if not model_path.exists():
        pytest.skip(f"Missing model asset: {model_path}")

    torch = pytest.importorskip("torch")
    torch_instance = torch_model(model_path=model_path, **kwargs)
    torch_params = torch_instance.get_rest_pose(batch_size=2, dtype=torch.float32)
    with torch.no_grad():
        torch_vertices = torch.compile(torch_instance.forward_vertices, backend="eager", fullgraph=True)(**torch_params)
    assert torch_vertices.shape[-1] == 3

    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp

    jax_instance = jax_model(model_path=model_path, **kwargs)
    jax_params = jax_instance.get_rest_pose(batch_size=2, dtype=jnp.float32)
    jax_vertices = jax.jit(jax_instance.forward_vertices)(**jax_params)
    assert np.asarray(jax_vertices).shape[-1] == 3
