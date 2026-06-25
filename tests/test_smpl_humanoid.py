import numpy as np
import pytest

from body_models.registry import create_model
from body_models.robots.smpl_humanoid.constants import BODY_JOINTS, JOINT_NAMES, PARENTS
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


def test_smpl_humanoid_loads_mjcf_primitive_xml(tmp_path) -> None:
    xml_path = tmp_path / "smpl_humanoid.xml"
    xml_path.write_text(_smpl_humanoid_xml(), encoding="utf-8")

    model = SmplHumanoid(xml_path)
    params = model.get_rest_pose()

    assert model.link_names == [f"{name}_geom" for name in JOINT_NAMES]
    assert model.num_vertices > 0
    assert model.forward_vertices(**params).shape == (model.num_vertices, 3)


def test_smpl_humanoid_xml_requires_canonical_hierarchy(tmp_path) -> None:
    xml_path = tmp_path / "smpl_humanoid.xml"
    xml_path.write_text("<mujoco><worldbody><body name='Pelvis'/></worldbody></mujoco>", encoding="utf-8")

    with pytest.raises(ValueError, match="missing body names"):
        SmplHumanoid(xml_path)


def _smpl_humanoid_xml() -> str:
    children = {idx: [] for idx in range(len(JOINT_NAMES))}
    for joint_idx, parent in enumerate(PARENTS):
        if parent >= 0:
            children[parent].append(joint_idx)

    def body_xml(joint_idx: int, indent: str = "    ") -> str:
        name = JOINT_NAMES[joint_idx]
        child_xml = "".join(body_xml(child, indent + "  ") for child in children[joint_idx])
        return (
            f'{indent}<body name="{name}" pos="0 0 0">\n'
            f'{indent}  <geom name="{name}_geom" type="sphere" size="0.01"/>\n'
            f"{child_xml}"
            f"{indent}</body>\n"
        )

    roots = "".join(body_xml(joint_idx) for joint_idx, parent in enumerate(PARENTS) if parent < 0)
    return f"<mujoco>\n  <worldbody>\n{roots}  </worldbody>\n</mujoco>\n"
