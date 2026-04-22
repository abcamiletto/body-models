"""Interface contract tests for all model classes/backends."""

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ASSET_DIR = Path(__file__).parent / "assets"
MODELS = ["smpl", "smplx", "flame", "skel", "anny", "mhr", "soma"]
BACKENDS = ["torch", "numpy", "jax"]


def _get_model_file(model_name: str) -> Path:
    model_dir = ASSET_DIR / model_name / "model"
    if not model_dir.exists():
        return model_dir
    for ext in (".npz", ".pkl"):
        for f in model_dir.glob(f"*{ext}"):
            return f
    return model_dir


def _class_name(model_name: str) -> str:
    return model_name.upper() if model_name != "flame" else "FLAME"


def _build_model(model_name: str, backend: str) -> Any:
    if backend == "jax":
        pytest.importorskip("jax")
        pytest.importorskip("flax")

    module = import_module(f"body_models.{model_name}.{backend}")
    cls = getattr(module, _class_name(model_name))
    model_path = _get_model_file(model_name)

    kwargs: dict[str, Any] = {}
    if model_name == "skel":
        if not model_path.exists():
            pytest.skip(f"Model assets not found: {model_path}")
        kwargs["gender"] = "male"
        kwargs["model_path"] = model_path
    elif model_name in {"smpl", "smplx", "flame"}:
        if not model_path.exists():
            pytest.skip(f"Model assets not found: {model_path}")
        kwargs["model_path"] = model_path
    elif model_path.exists():
        kwargs["model_path"] = model_path

    return cls(**kwargs)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model_name", MODELS)
def test_model_interface_attributes(model_name: str, backend: str) -> None:
    model = _build_model(model_name, backend)

    assert isinstance(model.num_joints, int)
    assert model.num_joints > 0

    assert isinstance(model.num_vertices, int)
    assert model.num_vertices > 0

    assert isinstance(model.joint_names, list)
    assert len(model.joint_names) == model.num_joints
    assert all(isinstance(name, str) for name in model.joint_names)
    assert isinstance(model.parents, list)
    assert len(model.parents) == model.num_joints
    assert all(isinstance(parent, int) for parent in model.parents)

    params = model.get_rest_pose(batch_size=1)
    skeleton = model.forward_skeleton(**params)
    assert skeleton.ndim == 4
    assert skeleton.shape[-2:] == (4, 4)
    assert skeleton.shape[-3] == len(model.joint_names)
    assert skeleton.shape[-3] == model.num_joints

    assert model.faces.ndim == 2
    assert model.faces.shape[0] > 0

    assert model.rest_vertices.ndim == 2
    assert model.rest_vertices.shape[0] == model.num_vertices
    assert model.rest_vertices.shape[1] == 3


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model_name", MODELS)
def test_forward_skeleton_joint_indices_matches_full_output(model_name: str, backend: str) -> None:
    model = _build_model(model_name, backend)

    params = model.get_rest_pose(batch_size=1)
    full = model.forward_skeleton(**params)

    mid = min(1, model.num_joints - 1)
    joint_indices = [model.num_joints - 1, 0, mid]
    subset = model.forward_skeleton(**params, joint_indices=joint_indices)

    full_np = np.asarray(full)
    subset_np = np.asarray(subset)
    expected_np = full_np[..., joint_indices, :, :]

    assert subset_np.shape[-3] == len(joint_indices)
    np.testing.assert_allclose(subset_np[..., :3, :3], expected_np[..., :3, :3], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(subset_np[..., :3, 3], expected_np[..., :3, 3], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model_name", MODELS)
def test_viser_exports_match_model_outputs(model_name: str, backend: str) -> None:
    model = _build_model(model_name, backend)

    bind_params = model.get_rest_pose(batch_size=1)
    mesh = model.to_viser_skinned_mesh(bind_params)
    bones = model.to_viser_bones(**bind_params)

    assert set(mesh) == {"vertices", "faces", "bone_wxyzs", "bone_positions", "skin_weights"}
    assert set(bones) == {"bone_wxyzs", "bone_positions"}

    full_vertices = np.asarray(model.forward_vertices(**bind_params))[0]
    full_skeleton = np.asarray(model.forward_skeleton(**bind_params))[0]
    parents = np.asarray(model.parents, dtype=np.int64)
    local_skeleton = full_skeleton.copy()
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            local_skeleton[joint_index] = np.linalg.solve(full_skeleton[parent_index], full_skeleton[joint_index])

    np.testing.assert_allclose(mesh["vertices"], full_vertices, atol=1e-6, rtol=1e-6)
    assert mesh["vertices"].shape == (model.num_vertices, 3)
    assert mesh["faces"].ndim == 2
    assert mesh["faces"].shape[1] == 3
    assert mesh["skin_weights"].shape == (model.num_vertices, model.num_joints)
    assert bones["bone_positions"].shape == (model.num_joints, 3)
    assert bones["bone_wxyzs"].shape == (model.num_joints, 4)

    if np.asarray(model.faces).shape[1] == 4:
        assert mesh["faces"].shape[0] == model.faces.shape[0] * 2
    else:
        np.testing.assert_array_equal(mesh["faces"], np.asarray(model.faces))

    np.testing.assert_allclose(mesh["skin_weights"], np.asarray(model.skin_weights), atol=1e-6, rtol=1e-6)

    np.testing.assert_allclose(bones["bone_positions"], local_skeleton[:, :3, 3], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(mesh["bone_positions"], bones["bone_positions"], atol=1e-6, rtol=1e-6)

    bone_rotmats = Rotation.from_quat(bones["bone_wxyzs"][:, [1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(bone_rotmats, local_skeleton[:, :3, :3], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(mesh["bone_wxyzs"], bones["bone_wxyzs"], atol=1e-6, rtol=1e-6)
