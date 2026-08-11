import model_cases
import numpy as np
import pytest


def test_anny_torch_compile_on_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from body_models.anny.torch import ANNY

    model = ANNY().cuda()
    params = model.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    with torch.no_grad():
        vertices = torch.compile(model.forward_vertices, backend="eager")(**params)
    assert vertices.is_cuda


def test_points_torch_compile_and_jax_jit() -> None:
    torch = pytest.importorskip("torch")
    from body_models.smplx.torch import SMPLX as TorchSMPLX

    torch_model = TorchSMPLX(gender="neutral")
    mapping = np.zeros((2, torch_model.num_vertices), dtype=np.float32)
    mapping[0, 0] = 1.0
    mapping[1, 100] = 1.0
    torch_params = torch_model.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    torch_regressor = torch_model.prepare_point_regressor(mapping)
    with torch.no_grad():
        points = torch.compile(torch_model.forward_points, backend="eager")(
            **torch_params,
            point_regressor=torch_regressor,
        )
    assert points.shape == (2, 2, 3)

    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp

    from body_models.smplx.jax import SMPLX as JaxSMPLX

    jax_model = JaxSMPLX(gender="neutral")
    jax_params = jax_model.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    jax_regressor = jax_model.prepare_point_regressor(mapping)
    points = jax.jit(jax_model.forward_points)(
        **jax_params,
        point_regressor=jax_regressor,
    )
    assert points.shape == (2, 2, 3)


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.SKINNED_MODELS)
def test_skinned_torch_compile_and_jax_jit(name, model_class, kwargs) -> None:
    torch = pytest.importorskip("torch")
    torch_class = model_cases.backend_model_class(name, "torch")
    torch_instance = torch_class(**kwargs)
    torch_params = torch_instance.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    with torch.no_grad():
        torch_vertices = torch.compile(torch_instance.forward_vertices, backend="eager")(**torch_params)
    assert torch_vertices.shape[-1] == 3

    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp

    jax_class = model_cases.backend_model_class(name, "jax")
    jax_instance = jax_class(**kwargs)
    jax_params = jax_instance.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    jax_vertices = jax.jit(jax_instance.forward_vertices)(**jax_params)
    assert np.asarray(jax_vertices).shape[-1] == 3


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.RIGID_BODY_MODELS)
def test_rigid_body_torch_compile_and_jax_jit(name, model_class, kwargs) -> None:
    # forward_meshes returns Python mesh payloads, so compile the array-valued link transform primitive.
    torch = pytest.importorskip("torch")
    torch_class = model_cases.backend_model_class(name, "torch")
    torch_instance = torch_class(**kwargs)
    torch_params = torch_instance.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    with torch.no_grad():
        torch_links = torch.compile(torch_instance.forward_links, backend="eager", fullgraph=True)(**torch_params)
    assert torch_links.shape[-2:] == (4, 4)

    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp

    jax_class = model_cases.backend_model_class(name, "jax")
    jax_instance = jax_class(**kwargs)
    jax_params = jax_instance.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    jax_links = jax.jit(jax_instance.forward_links)(**jax_params)
    assert np.asarray(jax_links).shape[-2:] == (4, 4)
