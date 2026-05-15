import numpy as np
import pytest

import model_cases

LEADING_DIM_BATCH_SHAPES = [(), (2,), (2, 2, 2)]


@pytest.mark.parametrize(("name", "numpy_model", "torch_model", "jax_model", "kwargs"), model_cases.MODELS)
def test_torch_and_jax_match_numpy(name, numpy_model, torch_model, jax_model, kwargs) -> None:
    numpy_instance = numpy_model(**kwargs)
    numpy_params = numpy_instance.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    expected = numpy_instance.forward_vertices(**numpy_params)

    torch = pytest.importorskip("torch")
    torch_instance = torch_model(**kwargs)
    torch_params = torch_instance.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    with torch.no_grad():
        torch_vertices = torch_instance.forward_vertices(**torch_params)
    np.testing.assert_allclose(torch_vertices.numpy(), expected, rtol=1e-4, atol=1e-4)

    pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp

    jax_instance = jax_model(**kwargs)
    jax_params = jax_instance.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    jax_vertices = jax_instance.forward_vertices(**jax_params)
    np.testing.assert_allclose(np.asarray(jax_vertices), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("name", "numpy_model", "torch_model", "_jax_model", "kwargs"), model_cases.MODELS)
def test_kernels_match_default(name, numpy_model, torch_model, _jax_model, kwargs) -> None:
    numpy_instance = numpy_model(**kwargs)
    for kernel in getattr(numpy_instance, "kernels", ())[1:]:
        params = numpy_instance.get_rest_pose(batch_dims=(2,), dtype=np.float32)
        expected = numpy_instance.forward_vertices(**params)
        actual = numpy_model(kernel=kernel, **kwargs).forward_vertices(**params)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    torch = pytest.importorskip("torch")
    torch_instance = torch_model(**kwargs)
    for kernel in getattr(torch_instance, "kernels", ())[1:]:
        params = torch_instance.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
        with torch.no_grad():
            expected = torch_instance.forward_vertices(**params)
            actual = torch_model(kernel=kernel, **kwargs).forward_vertices(**params)
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("name", "numpy_model", "torch_model", "jax_model", "kwargs"), model_cases.MODELS)
def test_forward_accepts_arbitrary_leading_dimensions(
    name,
    numpy_model,
    torch_model,
    jax_model,
    kwargs,
) -> None:
    model = numpy_model(**kwargs)
    vertex_indices = list(range(min(8, model.num_vertices)))
    joint_indices = list(range(min(8, model.num_joints)))
    forward_kwargs = {"apply_correctives": False} if name == "soma" else {}
    for batch_shape in LEADING_DIM_BATCH_SHAPES:
        shaped_params = model.get_rest_pose(batch_dims=batch_shape)

        shaped_vertices = model.forward_vertices(**shaped_params, vertex_indices=vertex_indices, **forward_kwargs)
        shaped_skeleton = model.forward_skeleton(**shaped_params, joint_indices=joint_indices, **forward_kwargs)

        assert shaped_vertices.shape == (*batch_shape, len(vertex_indices), 3)
        assert shaped_skeleton.shape == (*batch_shape, len(joint_indices), 4, 4)

        entry_indices = np.ndindex(batch_shape) if batch_shape else [()]
        for entry_index in entry_indices:
            entry_params = {
                key: value[entry_index][None] if batch_shape else value[None] for key, value in shaped_params.items()
            }
            entry_vertices = model.forward_vertices(**entry_params, vertex_indices=vertex_indices, **forward_kwargs)[0]
            entry_skeleton = model.forward_skeleton(**entry_params, joint_indices=joint_indices, **forward_kwargs)[0]

            np.testing.assert_allclose(
                np.asarray(shaped_vertices[entry_index]),
                np.asarray(entry_vertices),
                atol=1e-6,
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(shaped_skeleton[entry_index]),
                np.asarray(entry_skeleton),
                atol=1e-6,
                rtol=1e-6,
            )
