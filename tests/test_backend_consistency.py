import numpy as np
import pytest

import model_cases

LEADING_DIM_BATCH_SHAPES = [(), (2,), (2, 2, 2)]


def mesh_vertices(meshes):
    return np.concatenate([np.asarray(mesh["vertices"]) for mesh in meshes], axis=-2)


@pytest.mark.parametrize(("name", "numpy_model", "torch_model", "jax_model", "kwargs"), model_cases.SKINNED_MODELS)
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


@pytest.mark.parametrize(("name", "numpy_model", "torch_model", "jax_model", "kwargs"), model_cases.RIGID_BODY_MODELS)
def test_rigid_body_meshes_match_numpy(name, numpy_model, torch_model, jax_model, kwargs) -> None:
    numpy_instance = numpy_model(**kwargs)
    numpy_params = numpy_instance.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    expected_meshes = numpy_instance.forward_meshes(**numpy_params)
    expected = mesh_vertices(expected_meshes)

    torch = pytest.importorskip("torch")
    torch_instance = torch_model(**kwargs)
    torch_params = torch_instance.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    with torch.no_grad():
        torch_meshes = torch_instance.forward_meshes(**torch_params)
    assert [mesh["name"] for mesh in torch_meshes] == [mesh["name"] for mesh in expected_meshes]
    np.testing.assert_allclose(mesh_vertices(torch_meshes), expected, rtol=1e-4, atol=1e-4)

    pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp

    jax_instance = jax_model(**kwargs)
    jax_params = jax_instance.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    jax_meshes = jax_instance.forward_meshes(**jax_params)
    assert [mesh["name"] for mesh in jax_meshes] == [mesh["name"] for mesh in expected_meshes]
    np.testing.assert_allclose(mesh_vertices(jax_meshes), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("name", "numpy_model", "_torch_model", "_jax_model", "kwargs"), model_cases.RIGID_BODY_MODELS)
def test_rigid_body_models_do_not_expose_forward_vertices(name, numpy_model, _torch_model, _jax_model, kwargs) -> None:
    model = numpy_model(**kwargs)
    assert not hasattr(model, "forward_vertices")


@pytest.mark.parametrize(("name", "numpy_model", "torch_model", "_jax_model", "kwargs"), model_cases.SKINNED_MODELS)
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


@pytest.mark.parametrize(("name", "numpy_model", "torch_model", "jax_model", "kwargs"), model_cases.SKINNED_MODELS)
def test_skinned_forward_accepts_arbitrary_leading_dimensions(
    name,
    numpy_model,
    torch_model,
    jax_model,
    kwargs,
) -> None:
    model = numpy_model(**kwargs)
    vertex_indices = list(range(min(8, model.num_vertices)))
    joint_indices = list(range(min(8, model.num_joints)))
    for batch_shape in LEADING_DIM_BATCH_SHAPES:
        shaped_params = model.get_rest_pose(batch_dims=batch_shape)

        shaped_vertices = model.forward_vertices(**shaped_params, vertex_indices=vertex_indices)
        shaped_skeleton = model.forward_skeleton(**shaped_params, joint_indices=joint_indices)

        assert shaped_vertices.shape == (*batch_shape, len(vertex_indices), 3)
        assert shaped_skeleton.shape == (*batch_shape, len(joint_indices), 4, 4)

        entry_indices = np.ndindex(batch_shape) if batch_shape else [()]
        for entry_index in entry_indices:
            entry_params = {
                key: value[entry_index][None] if batch_shape else value[None] for key, value in shaped_params.items()
            }
            entry_vertices = model.forward_vertices(**entry_params, vertex_indices=vertex_indices)[0]
            entry_skeleton = model.forward_skeleton(**entry_params, joint_indices=joint_indices)[0]

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


@pytest.mark.parametrize(("name", "numpy_model", "torch_model", "jax_model", "kwargs"), model_cases.RIGID_BODY_MODELS)
def test_rigid_body_forward_accepts_arbitrary_leading_dimensions(
    name,
    numpy_model,
    torch_model,
    jax_model,
    kwargs,
) -> None:
    model = numpy_model(**kwargs)
    link_indices = list(range(min(2, len(model.link_names))))
    joint_indices = list(range(min(8, model.num_joints)))
    for batch_shape in LEADING_DIM_BATCH_SHAPES:
        shaped_params = model.get_rest_pose(batch_dims=batch_shape)

        shaped_meshes = model.forward_meshes(**shaped_params, link_indices=link_indices)
        shaped_skeleton = model.forward_skeleton(**shaped_params, joint_indices=joint_indices)

        assert len(shaped_meshes) == len(link_indices)
        for mesh in shaped_meshes:
            assert mesh["vertices"].shape[:-2] == batch_shape
            assert mesh["vertices"].shape[-1] == 3
        assert shaped_skeleton.shape == (*batch_shape, len(joint_indices), 4, 4)

        entry_indices = np.ndindex(batch_shape) if batch_shape else [()]
        for entry_index in entry_indices:
            entry_params = {
                key: value[entry_index][None] if batch_shape else value[None] for key, value in shaped_params.items()
            }
            entry_meshes = model.forward_meshes(**entry_params, link_indices=link_indices)
            entry_skeleton = model.forward_skeleton(**entry_params, joint_indices=joint_indices)[0]

            for shaped_mesh, entry_mesh in zip(shaped_meshes, entry_meshes, strict=True):
                np.testing.assert_allclose(
                    np.asarray(shaped_mesh["vertices"][entry_index]),
                    np.asarray(entry_mesh["vertices"][0]),
                    atol=1e-6,
                    rtol=1e-6,
                )
            np.testing.assert_allclose(
                np.asarray(shaped_skeleton[entry_index]),
                np.asarray(entry_skeleton),
                atol=1e-6,
                rtol=1e-6,
            )
