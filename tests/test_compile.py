import numpy as np
import pytest

import model_cases


@pytest.mark.parametrize(("name", "_numpy_model", "torch_model", "jax_model", "kwargs"), model_cases.SKINNED_MODELS)
def test_skinned_torch_compile_and_jax_jit(name, _numpy_model, torch_model, jax_model, kwargs) -> None:
    torch = pytest.importorskip("torch")
    torch_instance = torch_model(**kwargs)
    torch_params = torch_instance.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    with torch.no_grad():
        compiled = torch.compile(torch_instance.forward_vertices, backend="eager", fullgraph=True)
        torch_vertices = compiled(torch_params)
        torch_prepared_vertices = compiled(torch_instance.prepare(torch_params))
    assert torch_vertices.shape[-1] == 3
    torch.testing.assert_close(torch_prepared_vertices, torch_vertices)

    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp

    jax_instance = jax_model(**kwargs)
    jax_params = jax_instance.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    compiled = jax.jit(jax_instance.forward_vertices)
    jax_vertices = compiled(jax_params)
    jax_prepared_vertices = compiled(jax_instance.prepare(jax_params))
    assert np.asarray(jax_vertices).shape[-1] == 3
    np.testing.assert_allclose(jax_prepared_vertices, jax_vertices, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("name", "_numpy_model", "torch_model", "jax_model", "kwargs"), model_cases.RIGID_BODY_MODELS)
def test_rigid_body_torch_compile_and_jax_jit(name, _numpy_model, torch_model, jax_model, kwargs) -> None:
    # forward_meshes returns Python mesh payloads, so compile the array-valued link transform primitive.
    torch = pytest.importorskip("torch")
    torch_instance = torch_model(**kwargs)
    torch_params = torch_instance.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    with torch.no_grad():
        torch_links = torch.compile(torch_instance.forward_links, backend="eager", fullgraph=True)(torch_params)
    assert torch_links.shape[-2:] == (4, 4)

    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp

    jax_instance = jax_model(**kwargs)
    jax_params = jax_instance.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    jax_links = jax.jit(jax_instance.forward_links)(jax_params)
    assert np.asarray(jax_links).shape[-2:] == (4, 4)
