import pytest
import numpy as np

import model_cases
from body_models.anny import numpy as anny_numpy
from body_models.anny import pose as anny_pose
from model_assets import get_model_file


@pytest.mark.parametrize(("name", "numpy_model", "_torch_model", "_jax_model", "kwargs"), model_cases.MODELS)
def test_model_loads(name, numpy_model, _torch_model, _jax_model, kwargs) -> None:
    numpy_model(**kwargs)


def test_anny_native_rig_pose_pack_round_trips() -> None:
    body_pose = np.zeros((2, 52, 3), dtype=np.float32)
    head_pose = np.zeros((2, 0, 3), dtype=np.float32)
    hand_pose = np.zeros((2, 0, 3), dtype=np.float32)
    global_rotation = np.zeros((2, 3), dtype=np.float32)

    pose = anny_pose.pack_pose(
        np,
        global_rotation,
        body_pose,
        head_pose,
        hand_pose,
    )
    unpacked_global, unpacked_body, unpacked_head, unpacked_hand = anny_pose.unpack_pose(np, pose)

    assert pose.shape == (2, 53, 3)
    np.testing.assert_array_equal(unpacked_global, global_rotation)
    np.testing.assert_array_equal(unpacked_body, body_pose)
    assert unpacked_head.shape == (2, 0, 3)
    assert unpacked_hand.shape == (2, 0, 3)


@pytest.mark.parametrize(("rig", "num_joints"), [("game_engine", 53), ("mixamo", 52)])
def test_anny_refreshed_rig_variants_load_and_pose(rig: str, num_joints: int) -> None:
    model_path = get_model_file("anny")
    rig_file = model_path / "data" / "mpfb2" / "rigs" / "standard" / f"rig.{rig}.json"
    if not rig_file.exists():
        pytest.skip(f"ANNY test asset does not include {rig_file.name}")

    model = anny_numpy.ANNY(model_path, rig=rig)
    params = model.get_rest_pose(batch_dims=(1,))
    skeleton = model.forward_skeleton(**params)

    assert model.num_joints == num_joints
    assert params["body_pose"].shape[-2] == num_joints - 1
    assert params["head_pose"].shape[-2] == 0
    assert params["hand_pose"].shape[-2] == 0
    assert skeleton.shape[-3:] == (num_joints, 4, 4)
