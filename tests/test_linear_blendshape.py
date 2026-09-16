"""Behavioral tests for the shared linear blendshape engine."""

import numpy as np
import pytest
from nanomanifold import SO3

from body_models import _linear_blendshape as linear
from body_models._rotations import VALID_ROTATION_TYPES
from body_models._runtime import NumpyRuntime

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("rotation_type", VALID_ROTATION_TYPES)
def test_pose_blocks_compose_across_rotation_representations(rotation_type) -> None:
    rng = np.random.default_rng(0)
    root = rng.normal(scale=0.1, size=(2, 3)).astype(np.float32)
    body = rng.normal(scale=0.1, size=(2, 3, 3)).astype(np.float32)
    hands = rng.normal(scale=0.1, size=(2, 2, 3)).astype(np.float32)

    encoded_root = SO3.convert(root, src="axis_angle", dst=rotation_type, xp=np)
    encoded_body = SO3.convert(body, src="axis_angle", dst=rotation_type, xp=np)
    actual = linear.assemble_pose_matrices(
        NumpyRuntime(),
        [linear.PoseBlock(encoded_body, rotation_type), linear.PoseBlock(hands, "axis_angle")],
        encoded_root,
        rotation_type,
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
        linear.assemble_pose_matrices(
            NumpyRuntime(),
            [
                linear.PoseBlock(np.zeros((2, 3, 3), dtype=np.float32), "axis_angle"),
                linear.PoseBlock(np.zeros((3, 2, 3), dtype=np.float32), "axis_angle"),
            ],
            None,
            "axis_angle",
        )


@pytest.mark.parametrize("rotation_type", VALID_ROTATION_TYPES)
def test_pose_blocks_compose_symmetric_rotation_means_without_reparameterizing(rotation_type) -> None:
    rng = np.random.default_rng(1)
    pose = rng.normal(scale=0.1, size=(2, 3, 3)).astype(np.float32)
    mean = rng.normal(scale=0.2, size=(3, 3)).astype(np.float32)
    encoded = SO3.convert(pose, src="axis_angle", dst=rotation_type, xp=np)
    half_mean_rotation = SO3.convert(mean / 2, src="axis_angle", dst="rotmat", xp=np)

    actual = linear.assemble_pose_matrices(
        NumpyRuntime(),
        [linear.PoseBlock(encoded, rotation_type, half_mean_rotation=half_mean_rotation)],
        None,
        rotation_type,
    )
    expected = half_mean_rotation @ SO3.convert(pose, src="axis_angle", dst="rotmat", xp=np) @ half_mean_rotation

    np.testing.assert_allclose(actual[:, 1:], expected, rtol=1e-5, atol=1e-5)


def test_sixd_pose_blocks_apply_rotation_means_without_axis_angle_conversion(monkeypatch) -> None:
    pose = SO3.convert(np.zeros((2, 3, 3), dtype=np.float32), src="axis_angle", dst="sixd", xp=np)
    half_mean = np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)).copy()
    calls = []
    convert = SO3.convert

    def record(value, *, src, dst, **kwargs):
        calls.append((src, dst))
        return convert(value, src=src, dst=dst, **kwargs)

    monkeypatch.setattr(linear.SO3, "convert", record)
    linear.assemble_pose_matrices(
        NumpyRuntime(),
        [linear.PoseBlock(pose, "sixd", half_mean_rotation=half_mean)],
        None,
        "sixd",
    )

    assert calls == [("sixd", "rotmat")]


def test_pose_blocks_reject_two_means() -> None:
    pose = np.zeros((2, 3, 3), dtype=np.float32)
    half_mean = np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3))

    with pytest.raises(ValueError, match="cannot combine"):
        linear.assemble_pose_matrices(
            NumpyRuntime(),
            [linear.PoseBlock(pose, "axis_angle", axis_angle_mean=pose[0], half_mean_rotation=half_mean)],
            None,
            "axis_angle",
        )
