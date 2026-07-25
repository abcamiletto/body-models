import numpy as np
import pytest

import model_cases


@pytest.mark.parametrize(("name", "_numpy_model", "torch_model", "jax_model", "kwargs"), model_cases.SKINNED_MODELS)
def test_skinned_torch_compile_and_jax_jit(name, _numpy_model, torch_model, jax_model, kwargs) -> None:
    torch = pytest.importorskip("torch")
    torch_instance = torch_model(**kwargs)
    torch_params = torch_instance.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    with torch.no_grad():
        torch_vertices = torch.compile(torch_instance.forward_vertices, backend="eager", fullgraph=True)(**torch_params)
        torch_bound, torch_pose = model_cases.bind_model(torch_instance, torch_params)
        torch_bound_vertices = torch.compile(torch_bound.forward_vertices, backend="eager", fullgraph=True)(
            **torch_pose
        )
    assert torch_vertices.shape[-1] == 3
    assert torch_bound_vertices.shape == torch_vertices.shape

    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp

    jax_instance = jax_model(**kwargs)
    jax_params = jax_instance.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    jax_vertices = jax.jit(jax_instance.forward_vertices)(**jax_params)
    jax_bound, jax_pose = model_cases.bind_model(jax_instance, jax_params)
    jax_bound_vertices = jax.jit(jax_bound.forward_vertices)(**jax_pose)
    assert np.asarray(jax_vertices).shape[-1] == 3
    assert np.asarray(jax_bound_vertices).shape == np.asarray(jax_vertices).shape


@pytest.mark.parametrize(("name", "_numpy_model", "torch_model", "jax_model", "kwargs"), model_cases.RIGID_BODY_MODELS)
def test_rigid_body_torch_compile_and_jax_jit(name, _numpy_model, torch_model, jax_model, kwargs) -> None:
    # forward_meshes returns Python mesh payloads, so compile the array-valued link transform primitive.
    torch = pytest.importorskip("torch")
    torch_instance = torch_model(**kwargs)
    torch_params = torch_instance.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    with torch.no_grad():
        torch_links = torch.compile(torch_instance.forward_links, backend="eager", fullgraph=True)(**torch_params)
    assert torch_links.shape[-2:] == (4, 4)

    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp

    jax_instance = jax_model(**kwargs)
    jax_params = jax_instance.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    jax_links = jax.jit(jax_instance.forward_links)(**jax_params)
    assert np.asarray(jax_links).shape[-2:] == (4, 4)
