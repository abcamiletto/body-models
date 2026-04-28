from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import torch

from body_models import g1
from body_models.g1 import core
from body_models.g1.io import G1_MESH_JOINT_MAP
from gradient_utils import prepare_params, sampled_gradcheck

pytestmark = pytest.mark.fast

ASSET_DIR = Path(__file__).parent / "assets" / "g1" / "model"
XML_PATH = ASSET_DIR / "xml" / "g1.xml"
MESH_DIR = ASSET_DIR / "meshes" / "g1"

if not XML_PATH.exists() or not MESH_DIR.exists():
    pytest.skip(f"G1 test assets not found at {ASSET_DIR}", allow_module_level=True)

G1_JOINT_NAMES = [
    "pelvis_skel",
    "left_hip_pitch_skel",
    "left_hip_roll_skel",
    "left_hip_yaw_skel",
    "left_knee_skel",
    "left_ankle_pitch_skel",
    "left_ankle_roll_skel",
    "left_toe_base",
    "right_hip_pitch_skel",
    "right_hip_roll_skel",
    "right_hip_yaw_skel",
    "right_knee_skel",
    "right_ankle_pitch_skel",
    "right_ankle_roll_skel",
    "right_toe_base",
    "waist_yaw_skel",
    "waist_roll_skel",
    "waist_pitch_skel",
    "left_shoulder_pitch_skel",
    "left_shoulder_roll_skel",
    "left_shoulder_yaw_skel",
    "left_elbow_skel",
    "left_wrist_roll_skel",
    "left_wrist_pitch_skel",
    "left_wrist_yaw_skel",
    "left_hand_roll_skel",
    "right_shoulder_pitch_skel",
    "right_shoulder_roll_skel",
    "right_shoulder_yaw_skel",
    "right_elbow_skel",
    "right_wrist_roll_skel",
    "right_wrist_pitch_skel",
    "right_wrist_yaw_skel",
    "right_hand_roll_skel",
]
G1_NUM_LINK_MESHES = sum(len(meshes) for meshes in G1_MESH_JOINT_MAP.values())
MUJOCO_TO_SOMA = np.array(core.MUJOCO_TO_KIMODO, dtype=np.float32)
GLOBAL_ROTATION_SHAPES = {
    "axis_angle": (2, 3),
    "quat": (2, 4),
    "sixd": (2, 6),
    "matrix": (2, 3, 3),
    "rotmat": (2, 3, 3),
    "hinge": (2, 3, 3),
}


def _backend(backend: str):
    if backend == "jax":
        pytest.importorskip("jax")
        pytest.importorskip("flax")
    if backend == "torch":
        pytest.importorskip("torch")
    module = import_module(f"body_models.g1.{backend}")
    return module.G1


def _array(backend: str, value):
    if backend == "torch":
        import torch

        return torch.tensor(value, dtype=torch.float32)
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.asarray(value, dtype=jnp.float32)
    return np.asarray(value, dtype=np.float32)


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _rot_x(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


# ============================================================================
# Metadata and forward tests
# ============================================================================


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_g1_metadata_matches_kimodo_skeleton(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR, rotation_type="rotmat")

    assert model.num_joints == 34
    assert model.joint_names == G1_JOINT_NAMES
    assert len(model.qpos_joint_indices) == 29
    assert not {
        "pelvis_skel",
        "left_toe_base",
        "right_toe_base",
        "left_hand_roll_skel",
        "right_hand_roll_skel",
    } & set(model.qpos_joint_names)
    assert model.parents[:8] == [-1, 0, 1, 2, 3, 4, 5, 6]
    assert model.parents[15:18] == [0, 15, 16]
    assert model.num_vertices > G1_NUM_LINK_MESHES * 3
    assert model.faces.ndim == 2
    assert model.faces.shape[0] > G1_NUM_LINK_MESHES
    assert model.faces.shape[1] == 3
    assert len(model.link_names) == G1_NUM_LINK_MESHES
    with pytest.raises(NotImplementedError, match="rigid articulated"):
        model.skin_weights


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("rotation_type", core.VALID_ROTATION_TYPES)
def test_get_rest_pose_global_rotation_matches_rotation_type(backend: str, rotation_type: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR, rotation_type=rotation_type)

    params = model.get_rest_pose(batch_size=2)
    skeleton = _to_numpy(model.forward_skeleton(**params))

    assert params["global_rotation"].shape == GLOBAL_ROTATION_SHAPES[rotation_type]
    np.testing.assert_allclose(skeleton[:, 0, :3, :3], np.broadcast_to(np.eye(3), (2, 3, 3)), atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_forward_skeleton_uses_xml_offsets_in_kimodo_coordinates(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR, rotation_type="rotmat")
    params = model.get_rest_pose(batch_size=1)
    params = {key: _array(backend, _to_numpy(value)) for key, value in params.items()}
    params["global_translation"] = _array(backend, [[10.0, 20.0, 30.0]])

    skeleton = _to_numpy(model.forward_skeleton(**params))

    np.testing.assert_allclose(skeleton[0, 0, :3, 3], [10.0, 20.0, 30.0], atol=1e-6)
    np.testing.assert_allclose(skeleton[0, 1, :3, 3], [10.064452, 19.8973, 30.0], atol=1e-6)

    subset = _to_numpy(model.forward_skeleton(**params, joint_indices=[1, 0]))
    np.testing.assert_allclose(subset[:, 0], skeleton[:, 1], atol=1e-6)
    np.testing.assert_allclose(subset[:, 1], skeleton[:, 0], atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_g1_convention_selects_output_coordinate_system(backend: str) -> None:
    soma_model = _backend(backend)(model_path=ASSET_DIR, rotation_type="rotmat")
    mujoco_model = _backend(backend)(model_path=ASSET_DIR, rotation_type="rotmat", convention="mujoco")
    soma_params = soma_model.get_rest_pose(batch_size=1)
    mujoco_params = mujoco_model.get_rest_pose(batch_size=1)
    soma_params = {key: _array(backend, _to_numpy(value)) for key, value in soma_params.items()}
    mujoco_params = {key: _array(backend, _to_numpy(value)) for key, value in mujoco_params.items()}
    mujoco_translation = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    mujoco_params["global_translation"] = _array(backend, mujoco_translation)
    soma_params["global_translation"] = _array(backend, mujoco_translation @ MUJOCO_TO_SOMA.T)

    soma_skeleton = _to_numpy(soma_model.forward_skeleton(**soma_params))
    mujoco_skeleton = _to_numpy(mujoco_model.forward_skeleton(**mujoco_params))
    soma_vertices = _to_numpy(soma_model.forward_vertices(**soma_params))
    mujoco_vertices = _to_numpy(mujoco_model.forward_vertices(**mujoco_params))

    np.testing.assert_allclose(soma_skeleton[..., :3, 3], mujoco_skeleton[..., :3, 3] @ MUJOCO_TO_SOMA.T, atol=1e-6)
    np.testing.assert_allclose(soma_vertices, mujoco_vertices @ MUJOCO_TO_SOMA.T, atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_g1_rejects_invalid_convention(backend: str) -> None:
    with pytest.raises(ValueError, match="Invalid convention"):
        _backend(backend)(model_path=ASSET_DIR, convention="z-up")


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_forward_vertices_rigidly_attaches_stl_links(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR, rotation_type="rotmat")
    params = model.get_rest_pose(batch_size=1)
    params = {key: _array(backend, _to_numpy(value)) for key, value in params.items()}
    params["global_translation"] = _array(backend, [[10.0, 20.0, 30.0]])

    vertices = _to_numpy(model.forward_vertices(**params))

    shifted_params = {**params, "global_translation": _array(backend, [[-1.0, 2.0, 0.5]])}
    shifted = _to_numpy(model.forward_vertices(**shifted_params))
    expected_delta = np.broadcast_to(np.array([[[-11.0, -18.0, -29.5]]], dtype=np.float32), shifted.shape)
    np.testing.assert_allclose(shifted - vertices, expected_delta, atol=1e-6)

    with pytest.raises(TypeError):
        model.forward_vertices(**params, return_per_link=True)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_forward_links_returns_stl_link_transforms(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR, rotation_type="rotmat")
    params = model.get_rest_pose(batch_size=1)
    params = {key: _array(backend, _to_numpy(value)) for key, value in params.items()}
    params["global_translation"] = _array(backend, [[10.0, 20.0, 30.0]])

    links = _to_numpy(model.forward_links(**params))
    skeleton = _to_numpy(model.forward_skeleton(**params))

    assert links.shape == (1, len(model.link_names), 4, 4)

    link_idx = model.link_names.index("torso_link.STL")
    joint_idx = model.link_joint_indices[link_idx]
    geom = np.eye(4, dtype=np.float32)
    geom[:3, :3] = _to_numpy(model.link_geom_rotations)[link_idx]
    geom[:3, 3] = _to_numpy(model.link_geom_positions)[link_idx]
    np.testing.assert_allclose(links[0, link_idx], skeleton[0, joint_idx] @ geom, atol=1e-6)

    vertices = _to_numpy(model.forward_vertices(**params))
    local_vertex = _to_numpy(model.link_mesh("torso_link.STL")["vertices"])[0]
    transformed_vertex = links[0, link_idx, :3, :3] @ local_vertex + links[0, link_idx, :3, 3]
    np.testing.assert_allclose(
        transformed_vertex,
        vertices[0, model.link_vertex_starts[link_idx]],
        atol=1e-6,
    )


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_link_mesh_access_returns_static_stl_chunks(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR, rotation_type="rotmat")
    meshes = model.joint_meshes("waist_pitch_skel")

    assert [mesh["name"] for mesh in meshes] == ["torso_link.STL", "logo_link.STL", "head_link.STL"]
    for mesh in meshes:
        link_idx = model.link_names.index(mesh["name"])
        vertex_start = model.link_vertex_starts[link_idx]
        vertex_count = model.link_vertex_counts[link_idx]
        face_start = model.link_face_starts[link_idx]
        face_count = model.link_face_counts[link_idx]

        assert mesh["joint_name"] == "waist_pitch_skel"
        assert mesh["joint_index"] == model.joint_names.index("waist_pitch_skel")
        assert _to_numpy(mesh["vertices"]).shape == (vertex_count, 3)
        np.testing.assert_array_equal(
            _to_numpy(mesh["faces"]),
            _to_numpy(model.faces)[face_start : face_start + face_count] - vertex_start,
        )

    link_mesh = model.link_mesh("torso_link.STL")
    np.testing.assert_allclose(_to_numpy(link_mesh["vertices"]), _to_numpy(meshes[0]["vertices"]), atol=1e-6)
    np.testing.assert_array_equal(_to_numpy(link_mesh["faces"]), _to_numpy(meshes[0]["faces"]))


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_scalar_pose_uses_xml_hinge_axes(backend: str) -> None:
    hinge_model = _backend(backend)(model_path=ASSET_DIR, rotation_type="hinge")
    rotmat_model = _backend(backend)(model_path=ASSET_DIR, rotation_type="rotmat")
    left_hip_qpos = hinge_model.qpos_joint_names.index("left_hip_pitch_skel")
    scalar_pose = np.zeros((1, len(hinge_model.qpos_joint_indices), 1), dtype=np.float32)
    rotmat_pose = np.tile(np.eye(3, dtype=np.float32), (1, len(hinge_model.qpos_joint_indices), 1, 1))
    scalar_pose[0, left_hip_qpos, 0] = 0.7
    rotmat_pose[0, left_hip_qpos] = _rot_x(0.7)
    global_translation = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    rest_pose = hinge_model.get_rest_pose(batch_size=1)

    assert rest_pose["body_pose"].shape == (1, len(hinge_model.qpos_joint_indices), 1)
    assert rest_pose["global_rotation"].shape == (1, 3, 3)
    with pytest.raises(ValueError, match="body_pose"):
        rotmat_model.forward_skeleton(body_pose=_array(backend, scalar_pose))

    scalar_skeleton = _to_numpy(
        hinge_model.forward_skeleton(
            body_pose=_array(backend, scalar_pose),
            global_translation=_array(backend, global_translation),
        )
    )
    rotmat_skeleton = _to_numpy(
        rotmat_model.forward_skeleton(
            body_pose=_array(backend, rotmat_pose),
            global_translation=_array(backend, global_translation),
        )
    )
    np.testing.assert_allclose(scalar_skeleton, rotmat_skeleton, atol=1e-6)

    qpos = _to_numpy(
        g1.to_mujoco_qpos(
            hinge_model,
            body_pose=_array(backend, scalar_pose),
            global_translation=_array(backend, global_translation),
            clamp_to_limits=False,
        )
    )
    np.testing.assert_allclose(qpos[0, 7 + left_hip_qpos], 0.7, atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_to_mujoco_qpos_uses_xml_axis_limits_and_coordinate_transform(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR, rotation_type="rotmat")
    body_pose = np.tile(np.eye(3, dtype=np.float32), (1, len(model.qpos_joint_indices), 1, 1))
    left_hip_qpos = model.qpos_joint_names.index("left_hip_pitch_skel")
    body_pose[0, left_hip_qpos] = _rot_x(3.0)

    qpos = _to_numpy(
        g1.to_mujoco_qpos(
            model,
            body_pose=_array(backend, body_pose),
            global_translation=_array(backend, [[1.0, 2.0, 3.0]]),
            clamp_to_limits=True,
        )
    )

    assert qpos.shape == (1, 7 + len(model.qpos_joint_indices))
    np.testing.assert_allclose(qpos[0, :3], [3.0, 1.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(qpos[0, 3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(qpos[0, 7 + left_hip_qpos], 2.8798, atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_to_mujoco_qpos_preserves_mujoco_convention_root(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR, rotation_type="rotmat", convention="mujoco")
    body_pose = np.tile(np.eye(3, dtype=np.float32), (1, len(model.qpos_joint_indices), 1, 1))

    qpos = _to_numpy(
        g1.to_mujoco_qpos(
            model,
            body_pose=_array(backend, body_pose),
            global_translation=_array(backend, [[1.0, 2.0, 3.0]]),
        )
    )

    np.testing.assert_allclose(qpos[0, :3], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(qpos[0, 3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-6)


# ============================================================================
# Gradient tests (torch only)
# ============================================================================


@pytest.fixture
def model_float64():
    """Create G1 model in float64 for gradient checking."""
    from body_models.g1.torch import G1

    return G1(model_path=ASSET_DIR, rotation_type="rotmat").to(torch.float64).eval()


def test_gradients_forward_vertices(model_float64) -> None:
    """Test gradients flow correctly through forward_vertices."""
    params = prepare_params(model_float64.get_rest_pose(batch_size=1, dtype=torch.float64))
    inputs = tuple(params.values())

    def fn(*tensors):
        kwargs = dict(zip(params.keys(), tensors))
        return model_float64.forward_vertices(**kwargs)

    assert sampled_gradcheck(fn, inputs, n_samples=64)


def test_gradients_forward_skeleton(model_float64) -> None:
    """Test gradients flow correctly through forward_skeleton."""
    params = prepare_params(model_float64.get_rest_pose(batch_size=1, dtype=torch.float64))
    inputs = tuple(params.values())

    def fn(*tensors):
        kwargs = dict(zip(params.keys(), tensors))
        return model_float64.forward_skeleton(**kwargs)

    assert sampled_gradcheck(fn, inputs, n_samples=64)


def test_gradients_forward_links(model_float64) -> None:
    """Test gradients flow correctly through forward_links."""
    params = prepare_params(model_float64.get_rest_pose(batch_size=1, dtype=torch.float64))
    inputs = tuple(params.values())

    def fn(*tensors):
        kwargs = dict(zip(params.keys(), tensors))
        return model_float64.forward_links(**kwargs)

    assert sampled_gradcheck(fn, inputs, n_samples=64)
