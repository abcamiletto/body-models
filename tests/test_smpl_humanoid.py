import xml.etree.ElementTree as ET

import numpy as np
import pytest
from nanomanifold import SO3

from body_models.base import RigidBodyModel
from body_models.registry import create_model, list_models
from body_models.robots.smpl_humanoid.constants import BODY_JOINTS, JOINT_NAMES, PARENTS, SMPL_HUMANOID_VARIANTS
from body_models.robots.smpl_humanoid.io import SMPL_HUMANOID_SOURCES, get_model_path
from body_models.robots.smpl_humanoid.numpy import SmplHumanoid


@pytest.fixture
def smpl_humanoid_xml(tmp_path):
    xml_path = tmp_path / "smpl_humanoid.xml"
    xml_path.write_text(_smpl_humanoid_xml(), encoding="utf-8")
    return xml_path


def test_smpl_humanoid_factory_loads() -> None:
    model = create_model("smpl-humanoid")

    assert isinstance(model, SmplHumanoid)
    assert isinstance(model, RigidBodyModel)
    assert model.num_joints == 24
    assert len(model.forward_meshes(**model.get_rest_pose())) == 1


def test_smpl_humanoid_sources_are_variants() -> None:
    assert tuple(SMPL_HUMANOID_SOURCES) == SMPL_HUMANOID_VARIANTS


@pytest.mark.parametrize("source", sorted(SMPL_HUMANOID_SOURCES))
def test_smpl_humanoid_xml_uses_xyz_hinge_order(source: str) -> None:
    root = ET.parse(get_model_path(source)).getroot()
    bodies = {body.get("name"): body for body in root.findall(".//body")}
    for joint_name, _ in BODY_JOINTS:
        joints = [joint.get("name") for joint in bodies[joint_name].findall("joint")]
        assert joints == [f"{joint_name}_x", f"{joint_name}_y", f"{joint_name}_z"]


@pytest.mark.parametrize("model_name", sorted(SMPL_HUMANOID_VARIANTS))
def test_smpl_humanoid_variant_factories_load(model_name: str) -> None:
    model = create_model(model_name)

    assert model_name in list_models()
    assert isinstance(model, SmplHumanoid)
    assert model.num_joints == 24
    assert len(model.forward_meshes(**model.get_rest_pose())) == 1


@pytest.mark.parametrize("model_name", sorted(SMPL_HUMANOID_SOURCES))
def test_smpl_humanoid_variants_are_y_up(model_name: str) -> None:
    model = SmplHumanoid(model_name)
    assert_smpl_humanoid_is_y_up(model)


def test_smpl_humanoid_custom_xml_loads() -> None:
    xml_path = get_model_path("phc")
    model = SmplHumanoid(xml_path)

    assert_smpl_humanoid_is_y_up(model)


def test_smpl_humanoid_source_uses_config_path(smpl_humanoid_xml, monkeypatch) -> None:
    from body_models import config

    monkeypatch.setattr(
        config, "get_model_path", lambda model: smpl_humanoid_xml if model == "smpl-humanoid-phc" else None
    )

    assert get_model_path("phc") == smpl_humanoid_xml


def test_smpl_humanoid_custom_bare_xml_filename_loads(smpl_humanoid_xml, monkeypatch) -> None:
    monkeypatch.chdir(smpl_humanoid_xml.parent)

    model = SmplHumanoid(smpl_humanoid_xml.name)

    assert model.num_joints == 24


def assert_smpl_humanoid_is_y_up(model: SmplHumanoid) -> None:
    skeleton = model.forward_skeleton(**model.get_rest_pose())
    joint_positions = skeleton[:, :3, 3]
    by_name = {name: i for i, name in enumerate(model.joint_names)}

    assert joint_positions[by_name["L_Ankle"], 1] < joint_positions[by_name["Pelvis"], 1]
    assert joint_positions[by_name["R_Ankle"], 1] < joint_positions[by_name["Pelvis"], 1]
    assert joint_positions[by_name["Head"], 1] > joint_positions[by_name["Pelvis"], 1]


def test_smpl_humanoid_to_qpos_uses_mujoco_joint_coordinates(smpl_humanoid_xml) -> None:
    model = SmplHumanoid(smpl_humanoid_xml)
    body_pose = np.arange(model.num_actuated, dtype=np.float32)

    qpos = model.to_qpos(body_pose)

    np.testing.assert_array_equal(qpos[:3], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(qpos[7:], body_pose)


def test_smpl_humanoid_from_smpl_motion_returns_mujoco_joint_coordinates(smpl_humanoid_xml) -> None:
    model = SmplHumanoid(smpl_humanoid_xml)
    smpl_body_pose = np.arange(2 * 23 * 3, dtype=np.float32).reshape(2, 23, 3) / 100
    global_translation = np.arange(6, dtype=np.float32).reshape(2, 3) / 10
    global_rotation = np.array([[0.1, -0.2, 0.3], [0.2, 0.1, -0.1]], dtype=np.float32)
    pelvis_rotation = np.array([[0.05, 0.02, -0.03], [-0.04, 0.01, 0.02]], dtype=np.float32)

    motion = model.from_smpl_motion(
        smpl_body_pose,
        global_translation,
        global_rotation=global_rotation,
        pelvis_rotation=pelvis_rotation,
    )

    ordered = np.stack([smpl_body_pose[:, smpl_index] for _, smpl_index in BODY_JOINTS], axis=1)
    expected_body_pose = SO3.conversions.from_axis_angle_to_euler(ordered, convention="XYZ", xp=np).reshape(
        2, model.num_actuated
    )
    expected_global_rotation = SO3.convert(
        SO3.multiply(
            SO3.convert(global_rotation, src="axis_angle", dst="quat", xp=np),
            SO3.convert(pelvis_rotation, src="axis_angle", dst="quat", xp=np),
            xp=np,
        ),
        src="quat",
        dst="axis_angle",
        xp=np,
    )

    np.testing.assert_allclose(motion["body_pose"], expected_body_pose, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(motion["global_translation"], global_translation)
    np.testing.assert_allclose(motion["global_rotation"], expected_global_rotation, rtol=1e-6, atol=1e-6)
    qpos = model.to_qpos(**motion)
    np.testing.assert_allclose(qpos[..., 7:], motion["body_pose"])


def test_smpl_humanoid_from_smpl_motion_matches_forward_euler_convention(smpl_humanoid_xml) -> None:
    model = SmplHumanoid(smpl_humanoid_xml)
    smpl_body_pose = np.zeros((1, 23, 3), dtype=np.float32)
    smpl_body_pose[:, 0] = np.array([0.3, -0.2, 0.4], dtype=np.float32)

    motion = model.from_smpl_motion(smpl_body_pose)

    expected = SO3.conversions.from_axis_angle_to_rotmat(smpl_body_pose[:, 0], xp=np)
    actual = SO3.conversions.from_euler_to_rotmat(motion["body_pose"][:, :3], convention="XYZ", xp=np)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_smpl_humanoid_forward_skeleton_matches_mujoco_qpos() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = SmplHumanoid("humenv")
    mj_model = mujoco.MjModel.from_xml_path(str(get_model_path("humenv")))
    data = mujoco.MjData(mj_model)
    body_pose = np.linspace(-0.2, 0.2, model.num_actuated, dtype=np.float32)[None]
    global_translation = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    global_rotation = np.array([[0.1, -0.2, 0.05]], dtype=np.float32)

    data.qpos[:] = model.to_qpos(body_pose, global_translation, global_rotation=global_rotation)[0]
    mujoco.mj_forward(mj_model, data)
    skeleton = model.forward_skeleton(body_pose, global_translation, global_rotation=global_rotation)

    for joint_index, name in enumerate(model.joint_names):
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        np.testing.assert_allclose(skeleton[0, joint_index, :3, 3], data.xpos[body_id], rtol=1e-6, atol=1e-6)


def test_smpl_humanoid_apose_is_canonical_smpl_order(smpl_humanoid_xml) -> None:
    model = SmplHumanoid(smpl_humanoid_xml)
    body_pose = model.unpack_pose(model.get_apose()["body_pose"])

    np.testing.assert_allclose(body_pose["L_Thorax"], np.array([0.0, 0.0, 0.45], dtype=np.float32), atol=1e-7)
    np.testing.assert_allclose(body_pose["R_Thorax"], np.array([0.0, 0.0, -0.45], dtype=np.float32), atol=1e-7)
    np.testing.assert_allclose(body_pose["L_Shoulder"], np.array([0.0, 0.0, 0.35], dtype=np.float32), atol=1e-7)
    np.testing.assert_allclose(body_pose["R_Shoulder"], np.array([0.0, 0.0, -0.35], dtype=np.float32), atol=1e-7)


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
