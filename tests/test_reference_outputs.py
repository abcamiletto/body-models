import json

import model_assets
import model_cases
import numpy as np
import pytest
from nanomanifold import SO3

from body_models.anny import _pose as anny_pose
from body_models.mhr import _pose as mhr_pose
from body_models.skel import _pose as skel_pose
from body_models.soma import _pose as soma_pose
from body_models.soma._preprocess import preprocess_model as preprocess_soma
from body_models.soma.numpy import SOMA


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.REFERENCE_MODELS)
def test_numpy_reference_vertices(name, model_class, kwargs) -> None:
    model = model_class(**kwargs)
    inputs = reference_inputs(name)
    vertices = model.forward_vertices(**inputs)
    if name == "mhr":
        vertices = vertices * 100
    expected = np.load(model_cases.ASSETS / name / "outputs/0/vertices.npy")

    np.testing.assert_allclose(vertices[0], expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.REFERENCE_MODELS)
def test_numpy_reference_skeleton(name, model_class, kwargs) -> None:
    model = model_class(**kwargs)
    inputs = reference_inputs(name)
    skeleton = model_cases.forward_skeleton(model, inputs)
    skeleton_outputs = {"anny": "bone_poses.npy", "mhr": "skeleton.npy", "skel": "joints.npy"}
    filename = skeleton_outputs.get(name, "joints.npy")
    expected = np.load(model_cases.ASSETS / name / "outputs/0" / filename)
    if name == "mhr":
        skeleton = mhr_native_skeleton(skeleton)
    if name == "skel":
        skeleton = skeleton[..., :3, 3]

    if name == "smplx":
        np.testing.assert_allclose(skeleton[0, :, :3, 3], expected[: skeleton.shape[1]], rtol=1e-4, atol=1e-4)
    elif name == "mhr":
        assert_mhr_skeleton_close(skeleton[0], expected)
    elif name == "skel" or expected.shape[-2:] == (4, 4):
        np.testing.assert_allclose(skeleton[0], expected, rtol=1e-4, atol=1e-4)
    else:
        np.testing.assert_allclose(skeleton[0, ..., :3, 3], expected, rtol=1e-4, atol=1e-4)


def test_soma_matches_upstream(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    upstream_soma = pytest.importorskip("soma")

    upstream_path = model_assets.get_model_file("soma") / "upstream"
    if not upstream_path.is_dir():
        pytest.skip("SOMA-X assets are not available")
    upstream = upstream_soma.SOMALayer(data_root=upstream_path, device="cpu", identity_model_type="soma", mode="dense")
    model = SOMA(model_path=preprocess_soma(upstream_path, tmp_path), model_type="soma", rotation_type="axis_angle")

    rng = np.random.default_rng(0)
    pose = rng.normal(scale=0.35, size=(2, 77, 3)).astype(np.float32)
    shape = rng.normal(scale=0.8, size=(2, 128)).astype(np.float32)
    with torch.no_grad():
        expected = upstream(poses=torch.from_numpy(pose), identity_coeffs=torch.from_numpy(shape))["vertices"].numpy()

    global_rotation, body_pose, head_pose, hand_pose = soma_pose.unpack_pose(np, pose)
    vertices = model.forward_vertices(body_pose, head_pose, hand_pose, shape=shape, global_rotation=global_rotation)

    np.testing.assert_allclose(vertices, expected, atol=1e-4)


def test_soma_lod_vertex_counts() -> None:
    for lod, num_vertices in {"mid": 18056, "low": 4505, "xlo": 612}.items():
        model = SOMA(model_type="soma", rotation_type="axis_angle", lod=lod)
        assert model.num_vertices == num_vertices


def reference_inputs(name: str) -> dict[str, np.ndarray]:
    data = json.loads((model_cases.ASSETS / name / "inputs/0.json").read_text())
    if name == "smpl":
        return {
            "shape": np.asarray(data["shape"], dtype=np.float32)[None],
            "body_pose": np.asarray(data["body_pose"], dtype=np.float32).reshape(1, 23, 3),
            "pelvis_rotation": np.asarray(data["global_orient"], dtype=np.float32)[None],
            "global_translation": np.asarray(data["transl"], dtype=np.float32)[None],
        }
    if name == "smplx":
        hands = np.asarray(data["left_hand_pose"] + data["right_hand_pose"], dtype=np.float32).reshape(1, 30, 3)
        head = np.asarray(data["jaw_pose"] + data["leye_pose"] + data["reye_pose"], dtype=np.float32).reshape(1, 3, 3)
        return {
            "shape": np.asarray(data["shape"], dtype=np.float32)[None],
            "body_pose": np.asarray(data["body_pose"], dtype=np.float32).reshape(1, 21, 3),
            "hand_pose": hands,
            "head_pose": head,
            "expression": np.asarray(data["expression"], dtype=np.float32)[None],
            "pelvis_rotation": np.asarray(data["global_orient"], dtype=np.float32)[None],
            "global_translation": np.asarray(data["transl"], dtype=np.float32)[None],
        }
    if name == "flame":
        head_pose = data["neck_pose"] + data["jaw_pose"] + data["leye_pose"] + data["reye_pose"]
        return {
            "shape": np.asarray(data["shape"], dtype=np.float32)[None],
            "expression": np.asarray(data["expression"], dtype=np.float32)[None],
            "head_pose": np.asarray(head_pose, dtype=np.float32).reshape(1, 4, 3),
            "head_rotation": np.asarray(data["global_orient"], dtype=np.float32)[None],
            "global_translation": np.asarray(data["transl"], dtype=np.float32)[None],
        }
    if name == "skel":
        body_pose, head_pose = skel_pose.unpack_pose(np, np.asarray(data["body_pose"], dtype=np.float32)[None])
        return {
            "shape": np.asarray(data["shape"], dtype=np.float32)[None],
            "body_pose": body_pose,
            "head_pose": head_pose,
            "global_translation": np.asarray(data["trans"], dtype=np.float32)[None],
        }
    if name == "mhr":
        pose = np.asarray(data["pose"], dtype=np.float32)[None]
        body_pose, head_pose, hand_pose = mhr_pose.unpack_pose(np, pose)
        return {
            "shape": np.asarray(data["shape"], dtype=np.float32)[None],
            "body_pose": body_pose,
            "head_pose": head_pose,
            "hand_pose": hand_pose,
            "expression": np.asarray(data["expression"], dtype=np.float32)[None],
        }
    if name == "anny":
        phenotype = {key: np.asarray([value], dtype=np.float32) for key, value in data["phenotype"].items()}
        rotation = np.asarray(data["pose"], dtype=np.float32)[None, :, :3, :3]
        pose = SO3.convert(rotation, src="rotmat", dst="axis_angle", xp=np)
        global_rotation, body_pose, head_pose, hand_pose = anny_pose.unpack_pose(np, pose)
        return {
            "shape": np.stack(
                [
                    phenotype["gender"],
                    phenotype["age"],
                    phenotype["muscle"],
                    phenotype["weight"],
                    phenotype["height"],
                    phenotype["proportions"],
                ],
                axis=-1,
            ),
            "body_pose": body_pose,
            "head_pose": head_pose,
            "hand_pose": hand_pose,
            "global_rotation": global_rotation,
        }
    raise AssertionError(name)


def mhr_native_skeleton(skeleton: np.ndarray) -> np.ndarray:
    translation = skeleton[..., :3, 3] * 100
    rotation = skeleton[..., :3, :3]
    scale = np.linalg.vector_norm(rotation[..., :, 0], axis=-1, keepdims=True)
    quat = SO3.from_rotmat(rotation / scale[..., None], convention="xyzw", xp=np)
    return np.concatenate([translation, quat, scale], axis=-1)


def assert_mhr_skeleton_close(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual[:, :3], expected[:, :3], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual[:, 7:], expected[:, 7:], rtol=1e-4, atol=1e-4)
    quat_diff = np.minimum(np.abs(actual[:, 3:7] - expected[:, 3:7]), np.abs(actual[:, 3:7] + expected[:, 3:7]))
    np.testing.assert_allclose(quat_diff, 0.0, rtol=1e-4, atol=1e-4)
