"""Behavioral tests for the shared SMPL-family engine."""

import model_cases
import numpy as np
import pytest
from nanomanifold import SO3

from body_models import _smpl_family as family
from body_models._rotations import VALID_ROTATION_TYPES
from body_models.smpl import SMPL

SMPL_FAMILY_MODELS = [case for case in model_cases.MODELS if issubclass(case[1], family.SmplFamilyModel)]

pytestmark = pytest.mark.fast


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


def test_smpl_pose_corrective_joint_subset_matches_full_model() -> None:
    full = SMPL(gender="neutral")
    selected_name = "left_elbow"
    subset = SMPL(gender="neutral", pose_corrective_joints=[selected_name])
    without_correctives = SMPL(gender="neutral", pose_corrective_joints=[])
    params = full.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    params["body_pose"][..., full.joint_names.index(selected_name) - 1, :] = [0.2, -0.1, 0.3]

    expected = full.forward_vertices(**params, vertex_indices=range(32))
    actual = subset.forward_vertices(**params, vertex_indices=range(32))
    without_correctives_vertices = without_correctives.forward_vertices(**params, vertex_indices=range(32))
    _, full_pose = model_cases.prepare_states(full, params)
    disabled_identity, _ = model_cases.prepare_states(without_correctives, params)

    assert without_correctives_vertices.shape == (2, 32, 3)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(
        without_correctives.apply_pose_correctives(identity=disabled_identity, pose=full_pose),
        disabled_identity["rest_vertices"],
    )


@pytest.mark.parametrize("case", SMPL_FAMILY_MODELS, ids=lambda case: case[0])
def test_pose_corrective_lod_preserves_prepared_pose_contract(case) -> None:
    _, model_class, kwargs = case
    full = model_class(**kwargs)
    selected_name = full.pose_corrective_joint_names[0]
    selected = model_class(**kwargs, pose_corrective_joints=[selected_name])
    params = full.get_rest_pose(batch_dims=(2,), dtype=np.float32)
    for name, spec in full.parameter_spec.items():
        if spec.role == "pose":
            params[name][...] = 0.1

    _, full_pose = model_cases.prepare_states(full, params)
    _, selected_pose = model_cases.prepare_states(selected, params)
    full_basis = full.skinning_spec.corrective_basis
    selected_basis = selected.skinning_spec.corrective_basis
    joint_index = full.joint_names.index(selected_name)
    expected_indices = np.arange((joint_index - 1) * 9, joint_index * 9)

    assert selected.pose_corrective_joint_names == [selected_name]
    assert selected_basis.coefficient_dim == full_basis.coefficient_dim
    np.testing.assert_array_equal(selected_basis.coefficient_indices, expected_indices)
    np.testing.assert_array_equal(selected_basis.values, full_basis.values[:9])
    np.testing.assert_array_equal(full_pose["pose_coefficients"], selected_pose["pose_coefficients"])
