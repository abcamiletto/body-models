"""Shared linear deformation contracts."""

import numpy as np
import pytest

from body_models.common import deformation

pytestmark = pytest.mark.fast


def test_blend_shapes_supports_arbitrary_batch_dimensions() -> None:
    mean = np.arange(6, dtype=np.float32).reshape(2, 3)
    directions = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    coefficients = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    actual = deformation.blend_shapes(mean, directions, coefficients)
    expected = mean + np.einsum("...c,vdc->...vd", coefficients, directions)

    np.testing.assert_array_equal(actual, expected)


def test_pose_blend_shapes_excludes_the_root_rotation() -> None:
    rotations = np.broadcast_to(np.eye(3), (2, 3, 3, 3)).copy()
    rotations[..., 0, :, :] = 7.0
    rotations[..., 1, 0, 0] = 2.0
    directions = np.arange(18 * 6, dtype=np.float64).reshape(18, 6)

    actual = deformation.pose_blend_shapes(rotations, directions)
    features = (rotations[..., 1:, :, :] - np.eye(3)).reshape(2, -1)
    expected = (features @ directions).reshape(2, 2, 3)

    np.testing.assert_array_equal(actual, expected)
