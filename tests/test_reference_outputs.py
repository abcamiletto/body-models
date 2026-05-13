import json

import numpy as np
import pytest
from nanomanifold import SO3

import model_cases


@pytest.mark.parametrize(
    ("name", "numpy_model", "_torch_model", "_jax_model", "model_path", "kwargs"), model_cases.REFERENCE_MODELS
)
def test_numpy_reference_vertices(name, numpy_model, _torch_model, _jax_model, model_path, kwargs) -> None:
    model = numpy_model(model_path=model_path, **kwargs)
    inputs = reference_inputs(name)
    vertices = model.forward_vertices(**inputs)
    if name == "mhr":
        vertices = vertices * 100
    if name == "skel":
        vertices = vertices - model._feet_offset
    expected = np.load(model_cases.ASSETS / name / "outputs/0/vertices.npy")

    np.testing.assert_allclose(vertices[0], expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("name", "numpy_model", "_torch_model", "_jax_model", "model_path", "kwargs"), model_cases.REFERENCE_MODELS
)
def test_numpy_reference_skeleton(name, numpy_model, _torch_model, _jax_model, model_path, kwargs) -> None:
    model = numpy_model(model_path=model_path, **kwargs)
    inputs = reference_inputs(name)
    skeleton = model.forward_skeleton(**inputs)
    expected = reference_skeleton(name)
    if name == "mhr":
        skeleton = mhr_native_skeleton(skeleton)
    if name == "skel":
        skeleton = skeleton[..., :3, 3] - model._feet_offset

    if name == "smplx":
        np.testing.assert_allclose(skeleton[0, :, :3, 3], expected[: skeleton.shape[1]], rtol=1e-4, atol=1e-4)
    elif name == "mhr":
        assert_mhr_skeleton_close(skeleton[0], expected)
    elif name == "skel":
        np.testing.assert_allclose(skeleton[0], expected, rtol=1e-4, atol=1e-4)
    elif expected.shape[-2:] == (4, 4):
        np.testing.assert_allclose(skeleton[0], expected, rtol=1e-4, atol=1e-4)
    else:
        np.testing.assert_allclose(skeleton[0, ..., :3, 3], expected, rtol=1e-4, atol=1e-4)


def reference_inputs(name: str) -> dict[str, np.ndarray]:
    data = json.loads((model_cases.ASSETS / name / "inputs/0.json").read_text())
    if name == "smpl":
        return {
            "shape": array(data["shape"]),
            "body_pose": np.asarray(data["body_pose"], dtype=np.float32).reshape(1, 23, 3),
            "pelvis_rotation": array(data["global_orient"]),
            "global_translation": array(data["transl"]),
        }
    if name == "smplx":
        hands = np.asarray(data["left_hand_pose"] + data["right_hand_pose"], dtype=np.float32).reshape(1, 30, 3)
        head = np.asarray(data["jaw_pose"] + data["leye_pose"] + data["reye_pose"], dtype=np.float32).reshape(1, 3, 3)
        return {
            "shape": array(data["shape"]),
            "body_pose": np.asarray(data["body_pose"], dtype=np.float32).reshape(1, 21, 3),
            "hand_pose": hands,
            "head_pose": head,
            "expression": array(data["expression"]),
            "pelvis_rotation": array(data["global_orient"]),
            "global_translation": array(data["transl"]),
        }
    if name == "flame":
        pose = data["neck_pose"] + data["jaw_pose"] + data["leye_pose"] + data["reye_pose"]
        return {
            "shape": array(data["shape"]),
            "expression": array(data["expression"]),
            "pose": np.asarray(pose, dtype=np.float32).reshape(1, 4, 3),
            "head_rotation": array(data["global_orient"]),
            "global_translation": array(data["transl"]),
        }
    if name == "skel":
        return {
            "shape": array(data["shape"]),
            "pose": array(data["body_pose"]),
            "global_translation": array(data["trans"]),
        }
    if name == "mhr":
        return {"shape": array(data["shape"]), "pose": array(data["pose"]), "expression": array(data["expression"])}
    if name == "anny":
        phenotype = {key: np.asarray([value], dtype=np.float32) for key, value in data["phenotype"].items()}
        rotation = np.asarray(data["pose"], dtype=np.float32)[None, :, :3, :3]
        return {**phenotype, "pose": SO3.conversions.from_rotmat_to_axis_angle(rotation, xp=np)}
    raise AssertionError(name)


def reference_skeleton(name: str) -> np.ndarray:
    output_dir = model_cases.ASSETS / name / "outputs/0"
    for filename in ("skeleton.npy", "joints.npy", "bone_poses.npy"):
        path = output_dir / filename
        if path.exists():
            return np.load(path)
    raise AssertionError(f"Missing skeleton reference for {name}")


def array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)[None]


def mhr_native_skeleton(skeleton: np.ndarray) -> np.ndarray:
    translation = skeleton[..., :3, 3] * 100
    rotation = skeleton[..., :3, :3]
    scale = np.linalg.vector_norm(rotation[..., :, 0], axis=-1, keepdims=True)
    quat = SO3.conversions.from_rotmat_to_quat(rotation / scale[..., None], convention="xyzw", xp=np)
    return np.concatenate([translation, quat, scale], axis=-1)


def assert_mhr_skeleton_close(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual[:, :3], expected[:, :3], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual[:, 7:], expected[:, 7:], rtol=1e-4, atol=1e-4)
    quat_diff = np.minimum(np.abs(actual[:, 3:7] - expected[:, 3:7]), np.abs(actual[:, 3:7] + expected[:, 3:7]))
    np.testing.assert_allclose(quat_diff, 0.0, rtol=1e-4, atol=1e-4)
