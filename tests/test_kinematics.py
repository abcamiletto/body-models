import numpy as np
import pytest

from body_models._common.kinematics import (
    KinematicTree,
    affine_transforms,
    compute_sparse_skin_weights,
    invert_rigid_transforms,
    local_joint_offsets,
    rotation_between_vectors,
)

pytestmark = pytest.mark.fast

# SMPL's 24-joint kinematic tree (root at index 0, parent -1).
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def test_smpl_parents_every_joint_appears_once_with_correct_parent() -> None:
    fronts = KinematicTree.from_parents(SMPL_PARENTS).fronts

    seen: dict[int, int] = {}
    for joints, parents in fronts:
        for joint, parent in zip(joints, parents, strict=True):
            assert joint not in seen, f"joint {joint} appeared in more than one front"
            seen[joint] = parent

    assert sorted(seen) == list(range(len(SMPL_PARENTS)))
    for joint, parent in seen.items():
        expected = -1 if SMPL_PARENTS[joint] < 0 else SMPL_PARENTS[joint]
        assert parent == expected


def test_kinematic_tree_materializes_immutable_fronts() -> None:
    tree = KinematicTree.from_parents(SMPL_PARENTS)

    assert tree.parents == tuple(SMPL_PARENTS)
    assert tree.roots == (0,)
    hash(tree)


def test_forest_with_two_roots() -> None:
    # Two independent chains: 0 -> 1 -> 2, and 3 -> 4.
    parents = [-1, 0, 1, -1, 3]
    fronts = KinematicTree.from_parents(parents).fronts

    assert fronts[0] == ((0, 3), (-1, -1))
    assert fronts[1] == ((1, 4), (0, 3))
    assert fronts[2] == ((2,), (1,))


def test_parent_equal_joint_self_root() -> None:
    # SOMA convention: a root joint can be its own parent instead of -1.
    parents = [0, 0, 1]
    fronts = KinematicTree.from_parents(parents).fronts

    assert fronts[0] == ((0,), (-1,))
    assert fronts[1] == ((1,), (0,))
    assert fronts[2] == ((2,), (1,))


def test_cycle_raises_value_error() -> None:
    parents = [1, 0]
    with pytest.raises(ValueError, match="Invalid parent chain"):
        KinematicTree.from_parents(parents)


def test_kinematic_tree_select_prunes_to_ancestor_chains() -> None:
    tree = KinematicTree.from_parents(SMPL_PARENTS)

    selection = tree.select([22, 5])

    assert selection.cover_indices == (0, 2, 3, 5, 6, 9, 13, 16, 18, 20, 22)
    assert selection.output_indices == (10, 3)
    assert selection.tree.parents == (-1, 0, 0, 1, 2, 4, 5, 6, 7, 8, 9)
    assert tree.select([0]).cover_indices == (0,)
    empty = tree.select([])
    assert empty.cover_indices == ()
    assert empty.tree.roots == ()
    with pytest.raises(IndexError, match="joint_indices"):
        tree.select([len(SMPL_PARENTS)])


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


def test_affine_transforms_broadcasts_linear_and_translation_batches() -> None:
    linear = np.broadcast_to(np.eye(3), (3, 2, 3, 3))
    translation = np.arange(6).reshape(1, 2, 3)

    transforms = affine_transforms(linear, translation, xp=np)

    assert transforms.shape == (3, 2, 4, 4)
    np.testing.assert_array_equal(transforms[..., :3, 3], np.broadcast_to(translation, (3, 2, 3)))
    expected_bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (3, 2, 4))
    np.testing.assert_array_equal(transforms[..., 3, :], expected_bottom)


def test_invert_rigid_transforms() -> None:
    rotations = np.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    transforms = affine_transforms(rotations, np.array([[1.0, 2.0, 3.0]]), xp=np)

    inverse = invert_rigid_transforms(transforms, xp=np)

    np.testing.assert_allclose(transforms @ inverse, np.eye(4)[None])


def test_local_joint_offsets_preserves_each_root_position() -> None:
    joints = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])

    offsets = local_joint_offsets(joints, [-1, 0, 2], xp=np)

    expected = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    np.testing.assert_array_equal(offsets, expected)


def _assert_is_rotation(rotations: np.ndarray) -> None:
    identity = np.broadcast_to(np.eye(3), rotations.shape)
    np.testing.assert_allclose(rotations @ np.swapaxes(rotations, -1, -2), identity, atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-6)


def _rotate(rotations: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.squeeze(rotations @ vectors[..., None], axis=-1)


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def test_rotation_between_vectors_maps_source_onto_target() -> None:
    rng = np.random.default_rng(0)
    source = _unit(rng.normal(size=(16, 3)))
    target = _unit(rng.normal(size=(16, 3)))

    rotations = rotation_between_vectors(source, target, xp=np)

    _assert_is_rotation(rotations)
    np.testing.assert_allclose(_rotate(rotations, source), target, atol=1e-6)


def test_rotation_between_parallel_vectors_is_identity() -> None:
    source = _unit(np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 3.0]]))

    rotations = rotation_between_vectors(source, 2.0 * source, xp=np)

    np.testing.assert_allclose(rotations, np.broadcast_to(np.eye(3), (2, 3, 3)), atol=1e-6)


def test_rotation_between_antiparallel_vectors_is_a_half_turn() -> None:
    source = _unit(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))

    rotations = rotation_between_vectors(source, -source, xp=np)

    _assert_is_rotation(rotations)
    np.testing.assert_allclose(np.trace(rotations, axis1=-2, axis2=-1), -1.0, atol=1e-6)
    np.testing.assert_allclose(_rotate(rotations, source), -source, atol=1e-6)


def test_rotation_between_near_antiparallel_vectors_stays_a_rotation() -> None:
    angle = np.pi - 1e-4
    source = np.array([[1.0, 0.0, 0.0]])
    target = np.array([[np.cos(angle), np.sin(angle), 0.0]])

    rotations = rotation_between_vectors(source, target, xp=np)

    _assert_is_rotation(rotations)
    # Within 1e-4 rad of the pole the half-turn branch may pick any perpendicular
    # axis, so the image of the source is only accurate to the residual angle.
    np.testing.assert_allclose(_rotate(rotations, source), target, atol=1e-3)


def test_rotation_between_vectors_supports_extra_batch_dims() -> None:
    rng = np.random.default_rng(1)
    source = _unit(rng.normal(size=(2, 5, 3)))
    target = _unit(rng.normal(size=(2, 5, 3)))

    rotations = rotation_between_vectors(source, target, xp=np)

    assert rotations.shape == (2, 5, 3, 3)
    _assert_is_rotation(rotations)
    np.testing.assert_allclose(_rotate(rotations, source), target, atol=1e-6)
