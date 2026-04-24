from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import torch

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
def test_forward_vertices_rigidly_attaches_stl_links(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR, rotation_type="rotmat")
    params = model.get_rest_pose(batch_size=1)
    params = {key: _array(backend, _to_numpy(value)) for key, value in params.items()}
    params["global_translation"] = _array(backend, [[10.0, 20.0, 30.0]])

    vertices = _to_numpy(model.forward_vertices(**params))
    per_link = model.forward_vertices(**params, return_per_link=True)
    assert len(per_link) == G1_NUM_LINK_MESHES
    for link_idx, link in enumerate(per_link):
        start = model.link_vertex_starts[link_idx]
        count = model.link_vertex_counts[link_idx]
        assert link["name"] == model.link_names[link_idx]
        np.testing.assert_allclose(_to_numpy(link["vertices"])[0], vertices[0, start : start + count], atol=1e-6)

    shifted_params = {**params, "global_translation": _array(backend, [[-1.0, 2.0, 0.5]])}
    shifted = _to_numpy(model.forward_vertices(**shifted_params))
    expected_delta = np.broadcast_to(np.array([[[-11.0, -18.0, -29.5]]], dtype=np.float32), shifted.shape)
    np.testing.assert_allclose(shifted - vertices, expected_delta, atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_scalar_pose_uses_xml_hinge_axes(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR, rotation_type="rotmat")
    scalar_pose = np.zeros((1, model.num_joints, 1), dtype=np.float32)
    rotmat_pose = np.tile(np.eye(3, dtype=np.float32), (1, model.num_joints, 1, 1))
    scalar_pose[0, 1, 0] = 0.7
    rotmat_pose[0, 1] = _rot_x(0.7)
    global_translation = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)

    scalar_skeleton = _to_numpy(
        model.forward_skeleton(
            pose=_array(backend, scalar_pose),
            global_translation=_array(backend, global_translation),
        )
    )
    rotmat_skeleton = _to_numpy(
        model.forward_skeleton(
            pose=_array(backend, rotmat_pose),
            global_translation=_array(backend, global_translation),
        )
    )
    np.testing.assert_allclose(scalar_skeleton, rotmat_skeleton, atol=1e-6)

    qpos = _to_numpy(
        model.project_pose_to_qpos(
            pose=_array(backend, scalar_pose),
            global_translation=_array(backend, global_translation),
            clamp_to_limits=False,
        )
    )
    left_hip_qpos = model.qpos_joint_names.index("left_hip_pitch_skel")
    np.testing.assert_allclose(qpos[0, 7 + left_hip_qpos], 0.7, atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_project_pose_to_qpos_uses_xml_axis_limits_and_coordinate_transform(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR, rotation_type="rotmat")
    pose = np.tile(np.eye(3, dtype=np.float32), (1, model.num_joints, 1, 1))
    pose[0, 1] = _rot_x(3.0)

    qpos = _to_numpy(
        model.project_pose_to_qpos(
            pose=_array(backend, pose),
            global_translation=_array(backend, [[1.0, 2.0, 3.0]]),
            clamp_to_limits=True,
        )
    )

    assert qpos.shape == (1, 7 + len(model.qpos_joint_indices))
    np.testing.assert_allclose(qpos[0, :3], [3.0, 1.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(qpos[0, 3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    left_hip_qpos = model.qpos_joint_names.index("left_hip_pitch_skel")
    np.testing.assert_allclose(qpos[0, 7 + left_hip_qpos], 2.8798, atol=1e-6)


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
