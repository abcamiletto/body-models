import numpy as np
import pytest

from body_models.common.kinematics import compute_kinematic_fronts, compute_sparse_skin_weights

pytestmark = pytest.mark.fast

# SMPL's 24-joint kinematic tree (root at index 0, parent -1).
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def test_smpl_parents_first_front_is_root() -> None:
    fronts = compute_kinematic_fronts(SMPL_PARENTS)
    assert fronts[0] == ([0], [-1])


def test_smpl_parents_every_joint_appears_once_with_correct_parent() -> None:
    fronts = compute_kinematic_fronts(SMPL_PARENTS)

    seen: dict[int, int] = {}
    for joints, parents in fronts:
        for joint, parent in zip(joints, parents, strict=True):
            assert joint not in seen, f"joint {joint} appeared in more than one front"
            seen[joint] = parent

    assert sorted(seen) == list(range(len(SMPL_PARENTS)))
    for joint, parent in seen.items():
        expected = -1 if SMPL_PARENTS[joint] < 0 else SMPL_PARENTS[joint]
        assert parent == expected


def test_forest_with_two_roots() -> None:
    # Two independent chains: 0 -> 1 -> 2, and 3 -> 4.
    parents = [-1, 0, 1, -1, 3]
    fronts = compute_kinematic_fronts(parents)

    assert fronts[0] == ([0, 3], [-1, -1])
    assert fronts[1] == ([1, 4], [0, 3])
    assert fronts[2] == ([2], [1])


def test_parent_equal_joint_self_root() -> None:
    # SOMA convention: a root joint can be its own parent instead of -1.
    parents = [0, 0, 1]
    fronts = compute_kinematic_fronts(parents)

    assert fronts[0] == ([0], [-1])
    assert fronts[1] == ([1], [0])
    assert fronts[2] == ([2], [1])


def test_cycle_raises_value_error() -> None:
    parents = [1, 0]
    with pytest.raises(ValueError, match="Invalid parent chain"):
        compute_kinematic_fronts(parents)


def test_compute_sparse_skin_weights_reconstructs_dense_matrix() -> None:
    dense = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.25, 0.0, 0.25, 0.5],
        ],
        dtype=np.float32,
    )

    indices, weights = compute_sparse_skin_weights(dense)

    assert np.all(indices[weights == 0] == -1)
    reconstructed = np.zeros_like(dense)
    for vertex in range(dense.shape[0]):
        active = indices[vertex] >= 0
        np.add.at(reconstructed[vertex], indices[vertex, active], weights[vertex, active])

    np.testing.assert_allclose(reconstructed, dense)
