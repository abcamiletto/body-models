from importlib import import_module

import model_cases
import numpy as np
import pytest

from body_models._runtime import NumpyRuntime, TorchRuntime
from body_models.garment_measurements.numpy import GarmentMeasurements


class _RecordingRuntime(NumpyRuntime):
    kinematic_trees: list[tuple[int, ...]]

    def __init__(self) -> None:
        object.__setattr__(self, "kinematic_trees", [])

    def _compose_kinematic_tree(self, local_transforms, tree):
        assert len(tree.parents) == local_transforms.shape[-3]
        self.kinematic_trees.append(tuple(tree.parents))
        return super()._compose_kinematic_tree(local_transforms, tree)


LEADING_DIM_BATCH_SHAPES = [(), (2,), (2, 2, 2)]

_SHAPE_REQUIRED = "shape is required when identity is not provided"
_SHAPE_AND_EXPRESSION_REQUIRED = "shape and expression are required when identity is not provided"
MISSING_IDENTITY_ERRORS = {
    "anny": _SHAPE_REQUIRED,
    "flame": _SHAPE_AND_EXPRESSION_REQUIRED,
    "garment_measurements": _SHAPE_REQUIRED,
    "mano": _SHAPE_REQUIRED,
    "mhr": _SHAPE_AND_EXPRESSION_REQUIRED,
    "skel": _SHAPE_REQUIRED,
    "smpl": _SHAPE_REQUIRED,
    "smplh": _SHAPE_REQUIRED,
    "smplx": _SHAPE_AND_EXPRESSION_REQUIRED,
    "soma": _SHAPE_REQUIRED,
}


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_torch_and_jax_match_numpy(name, model_class, kwargs) -> None:
    numpy_instance = model_cases.backend_model_class(name, "numpy")(**kwargs)
    numpy_params = numpy_instance.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    expected = numpy_instance.forward_vertices(**numpy_params)

    torch = pytest.importorskip("torch")
    torch_instance = model_cases.backend_model_class(name, "torch")(**kwargs)
    torch_params = torch_instance.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    with torch.no_grad():
        torch_vertices = torch_instance.forward_vertices(**torch_params)
    np.testing.assert_allclose(torch_vertices.numpy(), expected, rtol=1e-4, atol=1e-4)

    pytest.importorskip("jax")
    import jax.numpy as jnp

    jax_instance = model_cases.backend_model_class(name, "jax")(**kwargs)
    jax_params = jax_instance.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    jax_vertices = jax_instance.forward_vertices(**jax_params)
    np.testing.assert_allclose(np.asarray(jax_vertices), expected, rtol=1e-4, atol=1e-4)


def test_garment_pelvis_rotation_defaults_to_identity() -> None:
    model = GarmentMeasurements()
    params = model.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    vertex_indices = np.arange(8)
    expected_vertices = model.forward_vertices(**params, vertex_indices=vertex_indices)
    expected_skeleton = model.forward_skeleton(**params, joint_indices=range(8))

    params.pop("pelvis_rotation")
    vertices = model.forward_vertices(**params, vertex_indices=vertex_indices)
    skeleton = model.forward_skeleton(**params, joint_indices=range(8))

    np.testing.assert_array_equal(vertices, expected_vertices)
    np.testing.assert_array_equal(skeleton, expected_skeleton)


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_kernel_backends_match_default(name, model_class, kwargs) -> None:
    torch = pytest.importorskip("torch")
    torch_class = model_cases.backend_model_class(name, "torch")
    torch_instance = torch_class(**kwargs)
    for kernel_backend in TorchRuntime.KERNEL_BACKENDS[1:]:
        params = torch_instance.get_rest_pose(batch_dims=(2, 2), dtype=torch.float32)
        vertex_indices = list(range(min(8, torch_instance.num_vertices)))
        with torch.no_grad():
            expected = torch_instance.forward_vertices(**params, vertex_indices=vertex_indices)
            model = torch_class(**kwargs, kernel_backend=kernel_backend)
            actual = model.forward_vertices(**params, vertex_indices=vertex_indices)
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_skinned_pose_uses_runtime_kinematics(name, model_class, kwargs) -> None:
    implementation_module = import_module(f"body_models.{name}._model")
    implementation_class = getattr(implementation_module, model_class.__name__)
    runtime = _RecordingRuntime()
    model = implementation_class(**kwargs, runtime=runtime)
    params = model.get_rest_pose()

    model.forward_skeleton(**params)
    assert runtime.kinematic_trees

    runtime.kinematic_trees.clear()
    model.forward_vertices(**params)
    assert runtime.kinematic_trees


@pytest.mark.parametrize(
    ("name", "model_class", "kwargs"),
    model_cases.MODELS,
)
def test_prepared_deformation_matches_forward(name, model_class, kwargs) -> None:
    from body_models._common import skinning

    def assert_compatible(model, params, xp):
        identity, pose = model_cases.prepare_states(model, params)
        spec = model.skinning_spec
        posed_vertices = model.apply_pose_correctives(identity=identity, pose=pose)
        vertices = skinning.linear_blend_skinning(
            posed_vertices,
            pose["skinning_transforms"],
            spec.skinning_weights,
            xp=xp,
        )
        prepared_params = model_cases.with_prepared_identity(model, params, identity)
        expected = model.forward_vertices(**prepared_params)
        np.testing.assert_allclose(np.asarray(vertices), np.asarray(expected), rtol=1e-4, atol=1e-4)

    numpy_instance = model_cases.backend_model_class(name, "numpy")(**kwargs)
    numpy_params = numpy_instance.get_rest_pose(dtype=np.float32)
    assert_compatible(numpy_instance, numpy_params, np)

    torch = pytest.importorskip("torch")
    torch_instance = model_cases.backend_model_class(name, "torch")(**kwargs)
    torch_params = torch_instance.get_rest_pose(dtype=torch.float32)
    with torch.no_grad():
        assert_compatible(torch_instance, torch_params, torch)

    pytest.importorskip("jax")
    import jax.numpy as jnp

    jax_instance = model_cases.backend_model_class(name, "jax")(**kwargs)
    jax_params = jax_instance.get_rest_pose(dtype=jnp.float32)
    assert_compatible(jax_instance, jax_params, jnp)


def test_raw_and_prepared_identity_are_mutually_exclusive() -> None:
    from body_models.smpl.numpy import SMPL

    model = SMPL(gender="neutral")
    params = model.get_rest_pose()
    identity = model.prepare_identity(params["shape"])

    with pytest.raises(ValueError, match="cannot be combined"):
        model.forward_vertices(**params, identity=identity)


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_raw_identity_coefficients_are_required(name, model_class, kwargs) -> None:
    model = model_class(**kwargs)
    params = model.get_rest_pose()
    pose_params = {key: value for key, value in params.items() if model.parameter_spec[key].role != "identity"}

    with pytest.raises(ValueError, match=MISSING_IDENTITY_ERRORS[name]):
        model.forward_vertices(**pose_params)


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_skinned_forward_accepts_arbitrary_leading_dimensions(
    name,
    model_class,
    kwargs,
) -> None:
    model = model_class(**kwargs)
    vertex_indices = list(range(min(8, model.num_vertices)))
    joint_indices = list(range(min(8, model.num_joints)))
    for batch_shape in LEADING_DIM_BATCH_SHAPES:
        shaped_params = model.get_rest_pose(batch_dims=batch_shape)

        shaped_vertices = model.forward_vertices(**shaped_params, vertex_indices=vertex_indices)
        shaped_skeleton = model_cases.forward_skeleton(model, shaped_params, joint_indices=joint_indices)

        assert shaped_vertices.shape == (*batch_shape, len(vertex_indices), 3)
        assert shaped_skeleton.shape == (*batch_shape, len(joint_indices), 4, 4)

        entry_indices = np.ndindex(batch_shape) if batch_shape else [()]
        for entry_index in entry_indices:
            entry_params = {
                key: value[entry_index][None] if batch_shape else value[None] for key, value in shaped_params.items()
            }
            entry_vertices = model.forward_vertices(**entry_params, vertex_indices=vertex_indices)[0]
            entry_skeleton = model_cases.forward_skeleton(model, entry_params, joint_indices=joint_indices)[0]

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


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_prepared_identity_broadcasts_across_pose_batch(
    name,
    model_class,
    kwargs,
) -> None:
    def assert_broadcasts(model, params):
        identity_params = {key: value[:1] for key, value in params.items()}
        identity, _ = model_cases.prepare_states(model, identity_params)
        vertex_indices = list(range(min(8, model.num_vertices)))
        joint_indices = list(range(min(8, model.num_joints)))

        expected_vertices = model.forward_vertices(**params, vertex_indices=vertex_indices)
        expected_skeleton = model_cases.forward_skeleton(model, params, joint_indices=joint_indices)
        prepared_params = model_cases.with_prepared_identity(model, params, identity)
        vertices = model.forward_vertices(**prepared_params, vertex_indices=vertex_indices)
        skeleton = model_cases.forward_skeleton(model, prepared_params, joint_indices=joint_indices)

        assert vertices.shape == (3, len(vertex_indices), 3)
        assert skeleton.shape == (3, len(joint_indices), 4, 4)
        np.testing.assert_allclose(np.asarray(vertices), np.asarray(expected_vertices), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(skeleton), np.asarray(expected_skeleton), rtol=1e-4, atol=1e-4)

    numpy_instance = model_cases.backend_model_class(name, "numpy")(**kwargs)
    numpy_params = numpy_instance.get_rest_pose(batch_dims=(3,), dtype=np.float32)
    assert_broadcasts(numpy_instance, numpy_params)

    torch = pytest.importorskip("torch")
    torch_instance = model_cases.backend_model_class(name, "torch")(**kwargs)
    torch_params = torch_instance.get_rest_pose(batch_dims=(3,), dtype=torch.float32)
    with torch.no_grad():
        assert_broadcasts(torch_instance, torch_params)

    pytest.importorskip("jax")
    import jax.numpy as jnp

    jax_instance = model_cases.backend_model_class(name, "jax")(**kwargs)
    jax_params = jax_instance.get_rest_pose(batch_dims=(3,), dtype=jnp.float32)
    assert_broadcasts(jax_instance, jax_params)
