"""SMPL-X vertex-to-joint position regressors."""

import numpy as np
import pytest

from body_models.smplx import SMPLX

pytestmark = pytest.mark.fast


def test_regressed_joint_positions_match_regressing_posed_vertices() -> None:
    model = SMPLX(gender="neutral")
    rng = np.random.default_rng(0)
    mapping = rng.uniform(size=(4, model.num_vertices)).astype(np.float32)
    mapping /= mapping.sum(axis=1, keepdims=True)
    regressor = model.prepare_joint_regressor(mapping)
    params = model.get_rest_pose(batch_dims=(2,))
    params["body_pose"][:, 0, 2] = 0.2
    params["global_translation"][:] = [1.0, 2.0, 3.0]

    actual = model.forward_joint_positions(**params, joint_regressor=regressor)
    vertices = model.forward_vertices(**params)
    expected = np.einsum("kv,bvc->bkc", mapping, vertices)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_regressed_joint_positions_scale_translation_by_mapping_weight() -> None:
    model = SMPLX(gender="neutral")
    mapping = np.zeros((1, model.num_vertices), dtype=np.float32)
    mapping[0, 0] = 2.0
    regressor = model.prepare_joint_regressor(mapping)
    params = model.get_rest_pose(batch_dims=(1,))
    params["global_translation"][:] = [1.0, 2.0, 3.0]

    actual = model.forward_joint_positions(**params, joint_regressor=regressor)
    expected = np.einsum("kv,bvc->bkc", mapping, model.forward_vertices(**params))

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_prepare_joint_regressor_validates_mapping_shape() -> None:
    model = SMPLX(gender="neutral")

    with pytest.raises(ValueError, match=r"mapping must have shape \[K, 10475\]"):
        model.prepare_joint_regressor(np.zeros((2, 3), dtype=np.float32))
