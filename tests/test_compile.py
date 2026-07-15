import numpy as np
import pytest

import model_cases


def test_soma_cuda_graph_forward_vertices() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs require CUDA")

    from body_models.bodies.soma import torch as soma_torch

    model = soma_torch.SOMA(kernel="torch").cuda().eval()
    params = model.get_rest_pose(batch_dims=(2,))
    identity = model.prepare_identity(params.pop("shape")[:1])
    vertex_indices = [0, 2, 4]
    captured = model.capture_forward_vertices(
        **params,
        identity=identity,
        vertex_indices=vertex_indices,
    )

    params["body_pose"] = params["body_pose"] + 0.01
    with torch.no_grad():
        expected = model.forward_vertices(
            **params,
            identity=identity,
            vertex_indices=vertex_indices,
        )
        actual = captured(**params)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("name", "_numpy_model", "torch_model", "jax_model", "kwargs"), model_cases.SKINNED_MODELS)
def test_skinned_torch_compile_and_jax_jit(name, _numpy_model, torch_model, jax_model, kwargs) -> None:
    torch = pytest.importorskip("torch")
    torch_instance = torch_model(**kwargs)
    torch_params = torch_instance.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    with torch.no_grad():
        torch_vertices = torch.compile(torch_instance.forward_vertices, backend="eager", fullgraph=True)(**torch_params)
    assert torch_vertices.shape[-1] == 3

    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp

    jax_instance = jax_model(**kwargs)
    jax_params = jax_instance.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    jax_vertices = jax.jit(jax_instance.forward_vertices)(**jax_params)
    assert np.asarray(jax_vertices).shape[-1] == 3


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
