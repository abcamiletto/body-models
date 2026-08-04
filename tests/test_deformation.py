"""Shared linear deformation behavior."""

import numpy as np
import pytest

from body_models._common import deformation

pytestmark = pytest.mark.fast


def test_blend_shapes_supports_arbitrary_batch_dimensions() -> None:
    mean = np.arange(6, dtype=np.float32).reshape(2, 3)
    directions = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    coefficients = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    actual = deformation.blend_shapes(mean, directions, coefficients, xp=np)
    expected = mean + np.einsum("...c,vdc->...vd", coefficients, directions)

    np.testing.assert_array_equal(actual, expected)


def test_pose_correctives_exclude_the_root_rotation() -> None:
    rotations = np.broadcast_to(np.eye(3), (2, 3, 3, 3)).copy()
    rotations[..., 0, :, :] = 7.0
    rotations[..., 1, 0, 0] = 2.0
    directions = np.arange(18 * 6, dtype=np.float64).reshape(18, 6)

    coefficients = deformation.pose_coefficients(rotations, xp=np)
    actual = deformation.DenseCorrectiveBasis(directions).apply(coefficients)
    features = (rotations[..., 1:, :, :] - np.eye(3)).reshape(2, -1)
    expected = (features @ directions).reshape(2, 2, 3)

    np.testing.assert_array_equal(actual, expected)


def test_dense_corrective_basis_selects_coefficients() -> None:
    coefficients = np.arange(24, dtype=np.float32).reshape(2, 12)
    values = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    indices = np.array([0, 3, 7, 11])
    basis = deformation.DenseCorrectiveBasis(values, coefficient_indices=indices, source_coefficient_dim=12)

    actual = basis.apply(coefficients)
    expected = (coefficients[..., indices] @ values).reshape(2, 2, 3)

    assert basis.coefficient_dim == 12
    np.testing.assert_array_equal(actual, expected)


def test_dense_corrective_basis_rejects_wrong_input_dimension() -> None:
    values = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    indices = np.array([0, 3, 7, 11])
    basis = deformation.DenseCorrectiveBasis(values, coefficient_indices=indices, source_coefficient_dim=12)

    with pytest.raises(ValueError, match="Expected 12"):
        basis.apply(np.zeros((2, 4), dtype=np.float32))


def test_prepare_linear_identity_shares_coefficients_across_joints_and_vertices() -> None:
    vertex_template = np.zeros((1, 3), dtype=np.float32)
    vertex_directions = np.ones((1, 3, 1), dtype=np.float32)
    joint_template = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    joint_directions = np.ones((2, 3, 1), dtype=np.float32)

    identity = deformation.prepare_linear_identity(
        vertex_template=vertex_template,
        vertex_directions=vertex_directions,
        joint_template=joint_template,
        joint_directions=joint_directions,
        parents=[-1, 0],
        coefficients=np.array([2.0], dtype=np.float32),
        xp=np,
    )

    np.testing.assert_array_equal(identity["rest_vertices"], [[2.0, 2.0, 2.0]])
    np.testing.assert_array_equal(identity["rest_joints"], [[2.0, 2.0, 2.0], [3.0, 2.0, 2.0]])
    np.testing.assert_array_equal(identity["local_joint_offsets"], [[2.0, 2.0, 2.0], [1.0, 0.0, 0.0]])
