import numpy as np
import pytest

from body_models.base import RigidBodyModel
from body_models.registry import create_model
from body_models.robots.smpl_humanoid.constants import BODY_JOINTS, JOINT_NAMES, PARENTS
from body_models.robots.smpl_humanoid.numpy import SmplHumanoid


@pytest.fixture
def smpl_humanoid_xml(tmp_path):
    xml_path = tmp_path / "smpl_humanoid.xml"
    xml_path.write_text(_smpl_humanoid_xml(), encoding="utf-8")
    return xml_path


def test_smpl_humanoid_factory_loads(smpl_humanoid_xml) -> None:
    model = create_model("smpl-humanoid", model_path=smpl_humanoid_xml)

    assert isinstance(model, SmplHumanoid)
    assert isinstance(model, RigidBodyModel)
    assert model.num_joints == 24
    assert len(model.forward_meshes(**model.get_rest_pose())) == 1


def test_smpl_humanoid_pose_uses_reference_joint_order(smpl_humanoid_xml) -> None:
    model = SmplHumanoid(smpl_humanoid_xml)
    body_pose_by_smpl = np.arange(23 * 3, dtype=np.float32).reshape(23, 3)
    body_pose = np.concatenate([body_pose_by_smpl[smpl_index] for _, smpl_index in BODY_JOINTS])

    qpos = model.to_mujoco_qpos(body_pose)

    np.testing.assert_array_equal(qpos[:3], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(qpos[7:], body_pose)


def test_smpl_humanoid_apose_is_canonical_smpl_order(smpl_humanoid_xml) -> None:
    model = SmplHumanoid(smpl_humanoid_xml)
    body_pose = model.unpack_pose(model.get_apose()["body_pose"])

    np.testing.assert_array_equal(body_pose["L_Thorax"], np.array([0.0, 0.0, 0.45], dtype=np.float32))
    np.testing.assert_array_equal(body_pose["R_Thorax"], np.array([0.0, 0.0, -0.45], dtype=np.float32))
    np.testing.assert_array_equal(body_pose["L_Shoulder"], np.array([0.0, 0.0, 0.35], dtype=np.float32))
    np.testing.assert_array_equal(body_pose["R_Shoulder"], np.array([0.0, 0.0, -0.35], dtype=np.float32))


def test_smpl_humanoid_loads_mjcf_primitive_xml(smpl_humanoid_xml) -> None:
    model = SmplHumanoid(smpl_humanoid_xml)
    params = model.get_rest_pose()

    assert model.link_names == [f"{name}_geom" for name in JOINT_NAMES]
    assert model.num_vertices > 0
    mesh = model.forward_meshes(**params)[0]
    assert mesh.vertices.shape == (model.num_vertices, 3)


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
