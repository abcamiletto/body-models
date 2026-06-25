import numpy as np

from body_models.registry import create_model
from body_models.robots.smpl_humanoid.constants import BODY_JOINTS
from body_models.robots.smpl_humanoid.numpy import SmplHumanoid


def test_smpl_humanoid_factory_loads() -> None:
    model = create_model("smpl-humanoid")

    assert isinstance(model, SmplHumanoid)
    assert model.num_joints == 24
    assert model.forward_vertices(**model.get_rest_pose()).shape == (model.num_vertices, 3)


def test_smpl_humanoid_qpos_uses_reference_joint_order() -> None:
    model = SmplHumanoid()
    body_pose = np.arange(23 * 3, dtype=np.float32).reshape(23, 3)

    qpos = model.to_qpos(body_pose)
    expected_joint_pos = np.concatenate([body_pose[smpl_index] for _, smpl_index in BODY_JOINTS])

    np.testing.assert_array_equal(qpos[:3], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(qpos[7:], expected_joint_pos)


def test_smpl_humanoid_apose_is_canonical_smpl_order() -> None:
    body_pose = SmplHumanoid().get_apose()["body_pose"]

    assert body_pose.shape == (23, 3)
    np.testing.assert_array_equal(body_pose[12], np.array([0.0, 0.0, 0.45], dtype=np.float32))
    np.testing.assert_array_equal(body_pose[13], np.array([0.0, 0.0, -0.45], dtype=np.float32))
    np.testing.assert_array_equal(body_pose[15], np.array([0.0, 0.0, 0.35], dtype=np.float32))
    np.testing.assert_array_equal(body_pose[16], np.array([0.0, 0.0, -0.35], dtype=np.float32))
