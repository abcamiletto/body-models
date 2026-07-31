"""Behavioral tests for the shared SMPL-family engine."""

import numpy as np
import pytest
from nanomanifold import SO3

from body_models import _smpl_family as family
from body_models._rotations import VALID_ROTATION_TYPES


@pytest.mark.parametrize("rotation_type", VALID_ROTATION_TYPES)
def test_pose_blocks_compose_across_rotation_representations(rotation_type) -> None:
    rng = np.random.default_rng(0)
    root = rng.normal(scale=0.1, size=(2, 3)).astype(np.float32)
    body = rng.normal(scale=0.1, size=(2, 3, 3)).astype(np.float32)
    hands = rng.normal(scale=0.1, size=(2, 2, 3)).astype(np.float32)

    encoded_root = SO3.convert(root, src="axis_angle", dst=rotation_type, xp=np)
    encoded_body = SO3.convert(body, src="axis_angle", dst=rotation_type, xp=np)
    actual = family.assemble_pose_matrices(
        [(encoded_body, rotation_type), (hands, "axis_angle")],
        encoded_root,
        rotation_type,
        xp=np,
    )
    expected = SO3.convert(
        np.concatenate([root[:, None], body, hands], axis=1),
        src="axis_angle",
        dst="rotmat",
        xp=np,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_pose_blocks_reject_different_batch_shapes() -> None:
    with pytest.raises(ValueError, match="same batch shape"):
        family.assemble_pose_matrices(
            [
                (np.zeros((2, 3, 3), dtype=np.float32), "axis_angle"),
                (np.zeros((3, 2, 3), dtype=np.float32), "axis_angle"),
            ],
            None,
            "axis_angle",
            xp=np,
        )
