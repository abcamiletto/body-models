"""MHR control packing contracts."""

import numpy as np
import pytest

from body_models.bodies.mhr.pose import pack_pose


@pytest.mark.fast
@pytest.mark.parametrize(
    ("name", "sizes"),
    [
        ("body_pose", (95, 5, 104)),
        ("head_pose", (94, 5, 105)),
        ("hand_pose", (94, 6, 104 + 1)),
    ],
)
def test_pack_pose_validates_each_control_block(name: str, sizes: tuple[int, int, int]) -> None:
    body_size, head_size, hand_size = sizes
    with pytest.raises(ValueError, match=name):
        pack_pose(
            np,
            np.zeros(body_size, dtype=np.float32),
            np.zeros(head_size, dtype=np.float32),
            np.zeros(hand_size, dtype=np.float32),
        )
