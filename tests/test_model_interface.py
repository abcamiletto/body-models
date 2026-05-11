"""Interface contract tests for all model classes/backends."""

from collections.abc import Mapping
from importlib import import_module
from typing import Any

import numpy as np
import pytest
from nanomanifold import SO3

import model_assets
from body_models.constants import Joint

pytestmark = pytest.mark.fast

MODELS = [
    "smpl",
    "smplh",
    "mano",
    "smplx",
    "flame",
    "skel",
    "anny",
    "mhr",
    "soma",
    "garment_measurements",
    "g1",
    "myofullbody",
    "brainco",
]
BACKENDS = ["torch", "numpy", "jax"]


def _build_model(model_name: str, backend: str) -> Any:
    if backend == "jax":
        pytest.importorskip("jax")
        pytest.importorskip("flax")

    module = import_module(f"body_models.{model_name}.{backend}")
    cls = getattr(module, model_assets.CLASS_NAMES[model_name])
    model_path = model_assets.get_model_file(model_name)
    if not model_path.exists():
        pytest.skip(f"Model assets not found: {model_path}")

    kwargs: dict[str, Any] = {"model_path": model_path}
    if model_name == "skel":
        kwargs["gender"] = "male"
    if model_name == "brainco":
        kwargs["rotation_type"] = "hinge"

    return cls(**kwargs)


def _local_skeleton(model: Any, forward_kwargs: dict[str, Any]) -> np.ndarray:
    full_skeleton = np.asarray(model.forward_skeleton(**forward_kwargs))[0]
    local_skeleton = full_skeleton.copy()
    for joint_index, parent_index in enumerate(model.parents):
        if parent_index >= 0:
            local_skeleton[joint_index] = np.linalg.solve(full_skeleton[parent_index], full_skeleton[joint_index])
    return local_skeleton


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
    assert isinstance(model.common_joints, Mapping)
    assert all(isinstance(joint, Joint) for joint in model.common_joints)
    assert all(name in model.joint_names for name in model.common_joints.values())
    assert isinstance(model.parents, list)
    assert len(model.parents) == model.num_joints
    assert all(isinstance(parent, int) for parent in model.parents)
    if model.is_rigid_body:
        with pytest.raises(NotImplementedError, match="rigid articulated"):
            model.skin_weights
    else:
        assert model.skin_weights.shape == (model.num_vertices, model.num_joints)

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

    forward_kwargs = model.get_rest_pose(batch_size=1)
    if model.is_rigid_body:
        with pytest.raises(NotImplementedError, match="skin_weights"):
            model.to_viser_skinned_mesh(**forward_kwargs)
        bones = model.to_viser_bones(**forward_kwargs)
        local_skeleton = _local_skeleton(model, forward_kwargs)
        assert set(bones) == {"bone_wxyzs", "bone_positions"}
        assert bones["bone_positions"].shape == (model.num_joints, 3)
        assert bones["bone_wxyzs"].shape == (model.num_joints, 4)
        np.testing.assert_allclose(bones["bone_positions"], local_skeleton[:, :3, 3], atol=1e-6, rtol=1e-6)
        bone_rotmats = SO3.conversions.from_quat_to_rotmat(bones["bone_wxyzs"], convention="wxyz", xp=np)
        np.testing.assert_allclose(bone_rotmats, local_skeleton[:, :3, :3], atol=1e-6, rtol=1e-6)
        return

    mesh = model.to_viser_skinned_mesh(**forward_kwargs)
    bones = model.to_viser_bones(**forward_kwargs)

    assert set(mesh) == {"vertices", "faces", "bone_wxyzs", "bone_positions", "skin_weights"}
    assert set(bones) == {"bone_wxyzs", "bone_positions"}

    full_vertices = np.asarray(model.forward_vertices(**forward_kwargs))[0]
    local_skeleton = _local_skeleton(model, forward_kwargs)

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

    bone_rotmats = SO3.conversions.from_quat_to_rotmat(bones["bone_wxyzs"], convention="wxyz", xp=np)
    np.testing.assert_allclose(bone_rotmats, local_skeleton[:, :3, :3], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(mesh["bone_wxyzs"], bones["bone_wxyzs"], atol=1e-6, rtol=1e-6)
