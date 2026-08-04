"""Arbitrary vertex mappings on skinned models."""

import model_cases
import numpy as np
import pytest

from body_models.flame import FLAME
from body_models.mano import MANO
from body_models.smpl import SMPL
from body_models.smplh import SMPLH
from body_models.smplx import SMPLX

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.SKINNED_MODELS)
def test_points_match_mapped_vertices(name, model_class, kwargs) -> None:
    model = model_class(**kwargs)
    mapping = np.zeros((2, model.num_vertices), dtype=np.float32)
    mapping[0, 0] = 1.0
    mapping[1, model.num_vertices // 2] = 1.0
    regressor = model.prepare_point_regressor(mapping)
    params = model.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    pose_name = next(name for name, spec in model.parameter_spec.items() if spec.role == "pose")
    params[pose_name].reshape(-1)[0] = 0.1
    params["global_translation"][:] = [1.0, 2.0, 3.0]

    actual = model.forward_points(**params, point_regressor=regressor)
    expected = np.einsum("kv,bvc->bkc", mapping, model.forward_vertices(**params))

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_points_accept_prepared_identity() -> None:
    model = SMPLX(gender="neutral")
    mapping = np.zeros((1, model.num_vertices), dtype=np.float32)
    mapping[0, 0] = 1.0
    regressor = model.prepare_point_regressor(mapping)
    params = model.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    identity, _ = model_cases.prepare_states(model, params)
    prepared_params = model_cases.with_prepared_identity(model, params, identity)

    actual = model.forward_points(**prepared_params, point_regressor=regressor)
    expected = np.einsum("kv,bvc->bkc", mapping, model.forward_vertices(**params))

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_dense_point_mapping_matches_vertices() -> None:
    model = SMPLX(gender="neutral")
    rng = np.random.default_rng(0)
    mapping = rng.uniform(size=(2, model.num_vertices)).astype(np.float32)
    mapping /= mapping.sum(axis=1, keepdims=True)
    regressor = model.prepare_point_regressor(mapping)
    params = model.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    params["body_pose"][:, 0, 2] = 0.2

    actual = model.forward_points(**params, point_regressor=regressor)
    expected = np.einsum("kv,bvc->bkc", mapping, model.forward_vertices(**params))

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("model_class", "kwargs", "identity_widths"),
    [
        pytest.param(SMPL, {"gender": "neutral"}, {"shape": 11}, id="smpl"),
        pytest.param(SMPLH, {"gender": "neutral"}, {"shape": 11}, id="smplh"),
        pytest.param(
            SMPLX,
            {"gender": "neutral"},
            {"shape": 11, "expression": 12},
            id="smplx",
        ),
        pytest.param(MANO, {"side": "right"}, {"shape": 7}, id="mano"),
        pytest.param(FLAME, {}, {"shape": 11, "expression": 13}, id="flame"),
    ],
)
def test_smpl_family_points_accept_arbitrary_identity_widths(model_class, kwargs, identity_widths) -> None:
    model = model_class(**kwargs)
    mapping = np.zeros((2, model.num_vertices), dtype=np.float32)
    mapping[0, 0] = 1.0
    mapping[1, model.num_vertices // 2] = 1.0
    regressor = model.prepare_point_regressor(mapping)
    params = model.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    rng = np.random.default_rng(0)
    for name, width in identity_widths.items():
        params[name] = rng.normal(scale=0.01, size=(2, width)).astype(np.float32)

    actual = model.forward_points(**params, point_regressor=regressor)
    expected = np.einsum("kv,bvc->bkc", mapping, model.forward_vertices(**params))

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_point_backends_match_numpy() -> None:
    numpy_model = SMPLX(gender="neutral")
    mapping = np.zeros((2, numpy_model.num_vertices), dtype=np.float32)
    mapping[0, 0] = 1.0
    mapping[1, 100] = 1.0
    numpy_params = numpy_model.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    expected = numpy_model.forward_points(
        **numpy_params,
        point_regressor=numpy_model.prepare_point_regressor(mapping),
    )

    torch = pytest.importorskip("torch")
    torch_model = SMPLX(gender="neutral", runtime="torch")
    torch_params = torch_model.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    actual = torch_model.forward_points(
        **torch_params,
        point_regressor=torch_model.prepare_point_regressor(mapping),
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-5)

    pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp

    jax_model = SMPLX(gender="neutral", runtime="jax")
    jax_params = jax_model.get_rest_pose(batch_dims=(2,), dtype=jnp.float32)
    actual = jax_model.forward_points(
        **jax_params,
        point_regressor=jax_model.prepare_point_regressor(mapping),
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


def test_points_scale_translation_by_mapping_weight() -> None:
    model = SMPLX(gender="neutral")
    mapping = np.zeros((1, model.num_vertices), dtype=np.float32)
    mapping[0, 0] = 2.0
    regressor = model.prepare_point_regressor(mapping)
    params = model.get_rest_pose(batch_dims=(1,), dtype=np.float32)
    params["global_translation"][:] = [1.0, 2.0, 3.0]

    actual = model.forward_points(**params, point_regressor=regressor)
    expected = np.einsum("kv,bvc->bkc", mapping, model.forward_vertices(**params))

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_prepare_point_regressor_validates_mapping_shape() -> None:
    model = SMPLX(gender="neutral")

    with pytest.raises(ValueError, match=r"mapping must have shape \[K, 10475\]"):
        model.prepare_point_regressor(np.zeros((2, 3), dtype=np.float32))
