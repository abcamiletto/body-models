import numpy as np
import pytest

from body_models.robots.g1.io import get_model_path
from body_models.robots.g1.numpy import G1


def test_g1_to_qpos_accepts_rest_pose_motion_dict() -> None:
    model = G1()
    motion = model.get_rest_pose(batch_dims=(2,), dtype=np.float32)

    qpos = model.to_qpos(**motion)

    assert qpos.shape == (2, 7 + model.num_actuated)
    np.testing.assert_array_equal(qpos[:, 7:], motion["body_pose"])


def test_g1_forward_skeleton_matches_mujoco_fk() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = G1()
    mj_model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    data = mujoco.MjData(mj_model)
    body_pose = np.linspace(-0.1, 0.1, model.num_actuated, dtype=np.float32)[None]
    global_translation = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    global_rotation = np.array([[0.05, -0.1, 0.08]], dtype=np.float32)

    data.qpos[:] = model.to_qpos(body_pose, global_translation, global_rotation=global_rotation)[0]
    mujoco.mj_forward(mj_model, data)
    skeleton = model.forward_skeleton(body_pose, global_translation, global_rotation=global_rotation)
    mujoco_to_model = np.asarray(model.mujoco_to_model)

    for joint_index, joint_name in enumerate(model.joint_names):
        body_name = _g1_body_name(joint_name)
        if body_name is None:
            continue
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        expected = mujoco_to_model @ data.xpos[body_id]
        np.testing.assert_allclose(skeleton[0, joint_index, :3, 3], expected, rtol=1e-6, atol=1e-6)


def _g1_body_name(joint_name: str) -> str | None:
    if joint_name == "pelvis_skel":
        return "pelvis"
    if joint_name == "waist_pitch_skel":
        return "torso_link"
    if joint_name in {"left_toe_base", "right_toe_base", "left_hand_roll_skel", "right_hand_roll_skel"}:
        return None
    return f"{joint_name.removesuffix('_skel')}_link"
