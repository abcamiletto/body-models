"""Tests for the MyoFullBody musculoskeletal model."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from body_models import myofullbody

pytestmark = pytest.mark.fast

ASSET_DIR = Path(__file__).parent / "assets" / "myofullbody" / "model"
MAIN_XML = ASSET_DIR / "body" / "myofullbody.xml"

if not MAIN_XML.exists():
    pytest.skip(f"MyoFullBody test assets not found at {ASSET_DIR}", allow_module_level=True)

# Ground-truth shape derived from the upstream MJCF tree (101 bodies, 122 1-DoF
# joints split as 112 hinge + 10 slide, 102 link meshes).
EXPECTED_NUM_JOINTS = 101
EXPECTED_NUM_QPOS = 122
EXPECTED_NUM_HINGE = 112
EXPECTED_NUM_SLIDE = 10
EXPECTED_NUM_LINKS = 102


def _backend(backend: str):
    if backend == "jax":
        pytest.importorskip("jax")
        pytest.importorskip("flax")
    if backend == "torch":
        pytest.importorskip("torch")
    module = import_module(f"body_models.myofullbody.{backend}")
    return module.MyoFullBody


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


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_metadata_matches_upstream_mjcf(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR)

    assert model.num_joints == EXPECTED_NUM_JOINTS
    assert model.num_qpos == EXPECTED_NUM_QPOS
    assert len(model.qpos_joint_names) == EXPECTED_NUM_QPOS
    assert sum(t == "hinge" for t in model.qpos_joint_types) == EXPECTED_NUM_HINGE
    assert sum(t == "slide" for t in model.qpos_joint_types) == EXPECTED_NUM_SLIDE
    assert len(model.link_names) == EXPECTED_NUM_LINKS
    assert model.parents[0] == -1
    assert all(p < i for i, p in enumerate(model.parents) if p >= 0)
    assert "Full Body" in model.joint_names
    assert "humerus_r" in model.joint_names
    assert "femur_r" in model.joint_names
    assert model.faces.ndim == 2 and model.faces.shape[1] == 3
    assert model.num_vertices > EXPECTED_NUM_LINKS * 3

    with pytest.raises(NotImplementedError, match="rigid articulated"):
        model.skin_weights


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_get_rest_pose_returns_zeros(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR)
    params = model.get_rest_pose(batch_size=2)

    assert params["body_pose"].shape == (2, EXPECTED_NUM_QPOS)
    assert params["global_rotation"].shape == (2, 3)
    assert params["global_translation"].shape == (2, 3)
    np.testing.assert_array_equal(_to_numpy(params["body_pose"]), 0)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_forward_skeleton_root_uses_global_pose(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR)
    params = model.get_rest_pose(batch_size=1)
    params = {k: _array(backend, _to_numpy(v)) for k, v in params.items()}
    params["global_translation"] = _array(backend, [[10.0, 20.0, 30.0]])

    skeleton = _to_numpy(model.forward_skeleton(**params))

    np.testing.assert_allclose(skeleton[0, 0, :3, 3], [10.0, 20.0, 30.0], atol=1e-6)
    np.testing.assert_allclose(skeleton[0, 0, :3, :3], np.eye(3), atol=1e-6)

    # joint_indices subset
    subset = _to_numpy(model.forward_skeleton(**params, joint_indices=[5, 0]))
    np.testing.assert_allclose(subset[:, 1], skeleton[:, 0], atol=1e-6)
    np.testing.assert_allclose(subset[:, 0], skeleton[:, 5], atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_forward_vertices_translates_rigidly(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR)
    params = model.get_rest_pose(batch_size=1)
    params = {k: _array(backend, _to_numpy(v)) for k, v in params.items()}
    params["global_translation"] = _array(backend, [[10.0, 20.0, 30.0]])

    base = _to_numpy(model.forward_vertices(**params))
    shifted_params = {**params, "global_translation": _array(backend, [[-1.0, 2.0, 0.5]])}
    shifted = _to_numpy(model.forward_vertices(**shifted_params))

    delta = np.broadcast_to(np.array([[[-11.0, -18.0, -29.5]]], dtype=np.float32), shifted.shape)
    np.testing.assert_allclose(shifted - base, delta, atol=1e-5)

    # vertex_indices subset
    sub = _to_numpy(model.forward_vertices(**params, vertex_indices=[0, 100, 200]))
    assert sub.shape == (1, 3, 3)
    np.testing.assert_allclose(sub[0, 0], base[0, 0], atol=1e-6)
    np.testing.assert_allclose(sub[0, 1], base[0, 100], atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_hinge_qpos_rotates_distal_body(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR)
    params = model.get_rest_pose(batch_size=1)

    rest_skeleton = _to_numpy(
        model.forward_skeleton(**{k: _array(backend, _to_numpy(v)) for k, v in params.items()})
    )

    # Pick the right hip flexion DoF and rotate; tibia_r must move while pelvis stays put.
    qpos_idx = model.qpos_joint_names.index("hip_flexion_r")
    body_pose = _to_numpy(params["body_pose"]).copy()
    body_pose[0, qpos_idx] = 0.6
    posed_skeleton = _to_numpy(
        model.forward_skeleton(
            body_pose=_array(backend, body_pose),
            global_rotation=_array(backend, _to_numpy(params["global_rotation"])),
            global_translation=_array(backend, _to_numpy(params["global_translation"])),
        )
    )

    pelvis_idx = model.joint_names.index("pelvis")
    tibia_idx = model.joint_names.index("tibia_r")
    np.testing.assert_allclose(posed_skeleton[0, pelvis_idx], rest_skeleton[0, pelvis_idx], atol=1e-6)
    assert np.linalg.norm(posed_skeleton[0, tibia_idx, :3, 3] - rest_skeleton[0, tibia_idx, :3, 3]) > 0.05


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_forward_links_attaches_geom_offsets(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR)
    params = model.get_rest_pose(batch_size=1)
    params = {k: _array(backend, _to_numpy(v)) for k, v in params.items()}
    params["global_translation"] = _array(backend, [[1.0, 2.0, 3.0]])

    links = _to_numpy(model.forward_links(**params))
    skeleton = _to_numpy(model.forward_skeleton(**params))
    assert links.shape == (1, EXPECTED_NUM_LINKS, 4, 4)

    # Pick a deterministic link and verify it equals body_T @ geom_T.
    link_idx = model.link_names.index("r_femur")
    joint_idx = model.link_joint_indices[link_idx]
    geom = np.eye(4, dtype=np.float32)
    geom[:3, :3] = _to_numpy(model.link_geom_rotations)[link_idx]
    geom[:3, 3] = _to_numpy(model.link_geom_positions)[link_idx]
    np.testing.assert_allclose(links[0, link_idx], skeleton[0, joint_idx] @ geom, atol=1e-5)

    # link_mesh chunk should agree with the global vertex output for that link.
    vertices = _to_numpy(model.forward_vertices(**params))
    local_vertex = _to_numpy(model.link_mesh("r_femur")["vertices"])[0]
    transformed = links[0, link_idx, :3, :3] @ local_vertex + links[0, link_idx, :3, 3]
    np.testing.assert_allclose(
        transformed, vertices[0, model.link_vertex_starts[link_idx]], atol=1e-5
    )


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_link_and_joint_meshes_lookup(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR)

    head_meshes = model.joint_meshes("head")
    assert head_meshes
    for mesh in head_meshes:
        assert mesh["joint_name"] == "head"
        assert mesh["joint_index"] == model.joint_names.index("head")

    direct = model.link_mesh(head_meshes[0]["name"])
    np.testing.assert_allclose(_to_numpy(direct["vertices"]), _to_numpy(head_meshes[0]["vertices"]), atol=1e-7)


# ---------------------------------------------------------------------------
# MuJoCo qpos round-trip (numerical only; mujoco runtime not required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_to_from_mujoco_qpos_roundtrip(backend: str) -> None:
    model = _backend(backend)(model_path=ASSET_DIR)
    rng = np.random.default_rng(0)
    body_pose = rng.standard_normal((1, EXPECTED_NUM_QPOS)).astype(np.float32) * 0.1
    global_t = np.array([[1.5, 2.5, 3.5]], dtype=np.float32)
    global_r = np.array([[0.3, -0.4, 0.2]], dtype=np.float32)

    qpos = _to_numpy(
        myofullbody.to_mujoco_qpos(
            model,
            body_pose=_array(backend, body_pose),
            global_translation=_array(backend, global_t),
            global_rotation=_array(backend, global_r),
        )
    )
    assert qpos.shape == (1, 7 + EXPECTED_NUM_QPOS)

    recon = myofullbody.from_mujoco_qpos(_array(backend, qpos))
    np.testing.assert_allclose(_to_numpy(recon["body_pose"]), body_pose, atol=1e-5)
    np.testing.assert_allclose(_to_numpy(recon["global_translation"]), global_t, atol=1e-5)
    np.testing.assert_allclose(_to_numpy(recon["global_rotation"]), global_r, atol=1e-5)
