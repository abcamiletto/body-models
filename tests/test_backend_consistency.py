import numpy as np
import pytest
from trimesh import Trimesh

import model_cases

LEADING_DIM_BATCH_SHAPES = [(), (2,), (2, 2, 2)]


def mesh_vertices(meshes):
    return np.stack([np.asarray(mesh.vertices) for mesh in meshes], axis=0)


def assert_pose_helpers_round_trip(model, pose) -> None:
    pose_by_joint = model.unpack_pose(pose)
    assert list(pose_by_joint) == list(dict.fromkeys(model.actuated_joint_names))
    assert sum(value.shape[-1] for value in pose_by_joint.values()) == model.num_actuated
    for value in pose_by_joint.values():
        assert value.shape[:-1] == pose.shape[:-1]
    np.testing.assert_array_equal(np.asarray(model.pack_pose(pose_by_joint)), np.asarray(pose))


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
    assert all(isinstance(mesh, Trimesh) for mesh in expected_meshes)
    assert len(expected_meshes) == 2
    expected = mesh_vertices(expected_meshes)

    torch = pytest.importorskip("torch")
    torch_instance = torch_model(**kwargs)
    torch_params = torch_instance.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    assert_pose_helpers_round_trip(
        torch_instance, torch_params["hand_pose" if "hand_pose" in torch_params else "body_pose"]
    )
    with torch.no_grad():
        torch_meshes = torch_instance.forward_meshes(**torch_params)
    assert all(isinstance(mesh, Trimesh) for mesh in torch_meshes)
    assert len(torch_meshes) == 2
    np.testing.assert_allclose(mesh_vertices(torch_meshes), expected, rtol=1e-4, atol=1e-4)

    pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp

    jax_instance = jax_model(**kwargs)
    jax_params = jax_instance.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    assert_pose_helpers_round_trip(jax_instance, jax_params["hand_pose" if "hand_pose" in jax_params else "body_pose"])
    jax_meshes = jax_instance.forward_meshes(**jax_params)
    assert all(isinstance(mesh, Trimesh) for mesh in jax_meshes)
    assert len(jax_meshes) == 2
    np.testing.assert_allclose(mesh_vertices(jax_meshes), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("name", "numpy_model", "_torch_model", "_jax_model", "kwargs"), model_cases.RIGID_BODY_MODELS)
def test_rigid_body_models_do_not_expose_forward_vertices(name, numpy_model, _torch_model, _jax_model, kwargs) -> None:
    model = numpy_model(**kwargs)
    assert not hasattr(model, "forward_vertices")


@pytest.mark.parametrize(("name", "numpy_model", "_torch_model", "_jax_model", "kwargs"), model_cases.RIGID_BODY_MODELS)
def test_rigid_body_joint_name_spaces(name, numpy_model, _torch_model, _jax_model, kwargs) -> None:
    model = numpy_model(**kwargs)
    params = model.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    pose_name = "hand_pose" if "hand_pose" in params else "body_pose"
    skeleton = model.forward_skeleton(**params)

    assert len(model.joint_names) == model.num_joints
    assert skeleton.shape[-3] == len(model.joint_names)
    assert len(model.actuated_joint_names) == model.num_actuated
    assert len(model.actuated_joint_types) == model.num_actuated
    assert model.actuated_joint_limits.shape == (model.num_actuated, 2)
    assert params[pose_name].shape == (2, model.num_actuated)

    assert_pose_helpers_round_trip(model, params[pose_name])

    assert not hasattr(model, "qpos_joint_names")
    assert not hasattr(model, "qpos_joint_indices")
    assert not hasattr(model, "qpos_joint_axes")
    assert not hasattr(model, "qpos_joint_limits")
    assert not hasattr(model, "num_qpos")
    assert not hasattr(model, "actuated_joint_indices")
    assert not hasattr(model, "actuated_joint_axes")


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
    joint_indices = list(range(min(8, model.num_joints)))
    for batch_shape in LEADING_DIM_BATCH_SHAPES:
        shaped_params = model.get_rest_pose(batch_dims=batch_shape)

        shaped_links = model.forward_links(**shaped_params)
        shaped_skeleton = model.forward_skeleton(**shaped_params, joint_indices=joint_indices)

        assert shaped_links.shape == (*batch_shape, len(model.link_names), 4, 4)
        assert shaped_skeleton.shape == (*batch_shape, len(joint_indices), 4, 4)

        entry_indices = np.ndindex(batch_shape) if batch_shape else [()]
        for entry_index in entry_indices:
            entry_params = {
                key: value[entry_index][None] if batch_shape else value[None] for key, value in shaped_params.items()
            }
            entry_links = model.forward_links(**entry_params)
            entry_skeleton = model.forward_skeleton(**entry_params, joint_indices=joint_indices)[0]

            np.testing.assert_allclose(
                np.asarray(shaped_links[entry_index]),
                np.asarray(entry_links[0]),
                atol=1e-6,
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(shaped_skeleton[entry_index]),
                np.asarray(entry_skeleton),
                atol=1e-6,
                rtol=1e-6,
            )
