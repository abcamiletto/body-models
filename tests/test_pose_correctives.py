"""Host-side pose-corrective selection tests."""

import numpy as np
import pytest

from body_models._common import pose_correctives

pytestmark = pytest.mark.fast


def test_zero_blocks_are_pruned_from_the_default_selection() -> None:
    posedirs = np.arange(27 * 6, dtype=np.float32).reshape(27, 6)
    posedirs[9:18] = 0.0

    pruned, coefficient_indices, joint_names = pose_correctives.select_blocks(
        posedirs,
        ["root", "first", "zero", "third"],
        None,
    )

    assert joint_names == ("first", "third")
    np.testing.assert_array_equal(coefficient_indices, np.concatenate([np.arange(9), np.arange(18, 27)]))
    np.testing.assert_array_equal(pruned, np.concatenate([posedirs[:9], posedirs[18:]]))


def test_explicit_selection_uses_asset_order_and_preserves_zero_blocks() -> None:
    posedirs = np.zeros((27, 6), dtype=np.float32)
    posedirs[:9] = 1.0

    selected, coefficient_indices, selected_names = pose_correctives.select_blocks(
        posedirs,
        ["root", "first", "second", "third"],
        ["third", "second"],
    )

    assert selected.shape == (18, 6)
    assert selected_names == ("second", "third")
    np.testing.assert_array_equal(coefficient_indices, np.arange(9, 27))


@pytest.mark.parametrize(
    ("selected", "exception", "message"),
    [
        ("first", TypeError, "sequence"),
        ([1], TypeError, "only joint names"),
        (["first", "first"], ValueError, "duplicate"),
        (["root"], ValueError, "Unknown or root"),
        (["missing"], ValueError, "Unknown or root"),
    ],
)
def test_selection_rejects_invalid_names(selected, exception, message) -> None:
    with pytest.raises(exception, match=message):
        pose_correctives.select_blocks(
            np.ones((18, 3), dtype=np.float32),
            ["root", "first", "second"],
            selected,
        )
