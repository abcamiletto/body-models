import json

import numpy as np
import pytest
from nanomanifold import SO3

import model_assets
import model_cases
from body_models.anny import pose as anny_pose
from body_models.base import RigidBodyModel
from body_models.mhr import pose as mhr_pose
from body_models.skel import pose as skel_pose
from body_models.bodies.soma import pose as soma_pose
from body_models.bodies.soma.generate_asset import generate_asset as generate_soma_asset
from body_models.bodies.soma.numpy import SOMA


@pytest.mark.parametrize(("name", "numpy_model", "_torch_model", "_jax_model", "kwargs"), model_cases.REFERENCE_MODELS)
def test_numpy_reference_vertices(name, numpy_model, _torch_model, _jax_model, kwargs) -> None:
    model = numpy_model(**kwargs)
    inputs = reference_inputs(name)
    if isinstance(model, RigidBodyModel):
        vertices = np.stack([mesh.vertices for mesh in model.forward_meshes(**inputs)], axis=0)
    else:
        vertices = model.forward_vertices(**inputs)
    if name == "mhr":
        vertices = vertices * 100
    expected = np.load(model_cases.ASSETS / name / "outputs/0/vertices.npy")

    np.testing.assert_allclose(vertices[0], expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("name", "numpy_model", "_torch_model", "_jax_model", "kwargs"), model_cases.REFERENCE_MODELS)
def test_numpy_reference_skeleton(name, numpy_model, _torch_model, _jax_model, kwargs) -> None:
    model = numpy_model(**kwargs)
    inputs = reference_inputs(name)
    skeleton = model.forward_skeleton(**inputs)
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
    elif name == "skel":
        np.testing.assert_allclose(skeleton[0], expected, rtol=1e-4, atol=1e-4)
    elif expected.shape[-2:] == (4, 4):
        np.testing.assert_allclose(skeleton[0], expected, rtol=1e-4, atol=1e-4)
    else:
        np.testing.assert_allclose(skeleton[0, ..., :3, 3], expected, rtol=1e-4, atol=1e-4)


def test_soma_021_matches_upstream_pure_lbs(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    upstream_soma = pytest.importorskip("soma")

    model_path = model_assets.get_model_file("soma")
    upstream_model_path = tmp_path / "soma-upstream"
    upstream_model_path.mkdir()
    upstream_npz = model_path / "SOMA_neutral.upstream-0.2.1.npz"
    required_assets = [
        upstream_npz,
        model_path / "SOMA_template_rig.usda",
        model_path / "SOMA_procedural_transforms.json",
    ]
    package_assets = [model_path / "correctives_model.pt"]
    if not all(path.exists() for path in [*required_assets, *package_assets]):
        pytest.skip("SOMA 0.2.1 assets are not available")
    (upstream_model_path / "SOMA_neutral.npz").symlink_to(upstream_npz)
    for asset in [*required_assets[1:], *package_assets]:
        (upstream_model_path / asset.name).symlink_to(asset)

    upstream = upstream_soma.SOMALayer(
        data_root=upstream_model_path,
        device="cpu",
        identity_model_type="soma",
        mode="dense",
        lod="mid",
        correctives_model_path=None,
    )
    normalized_model_path = tmp_path / "soma-normalized"
    generate_soma_asset(upstream_model_path, normalized_model_path)
    model = SOMA(model_path=normalized_model_path, model_type="soma", rotation_type="axis_angle")

    shape = np.zeros((1, 128), dtype=np.float32)
    poses = np.zeros((3, 1, 77, 3), dtype=np.float32)
    poses[1, 0, 0, 0] = 0.4
    poses[2, 0, 38, 0] = 0.8

    for pose in poses:
        with torch.no_grad():
            expected = (
                upstream(
                    poses=torch.as_tensor(pose),
                    identity_coeffs=torch.zeros(1, 128),
                    apply_correctives=False,
                )["vertices"]
                .detach()
                .numpy()
            )

        global_rotation, body_pose, head_pose, hand_pose = soma_pose.unpack_pose(np, pose)
        identity = model.prepare_identity(shape)
        prepared_pose = model.prepare_pose(body_pose, head_pose, hand_pose, global_rotation, identity=identity)
        vertices = model._kernel.forward_vertices(
            data=model.weights,
            global_translation=None,
            vertex_indices=None,
            rotation_type=model.rotation_type,
            rest_vertices=identity["rest_vertices"],
            skinning_transforms=prepared_pose["skinning_transforms"],
            pose_offsets=np.zeros_like(prepared_pose["pose_offsets"]),
            xp=np,
        )

        np.testing.assert_allclose(vertices, expected, rtol=2e-3, atol=2e-3)


def test_soma_lod_vertex_counts() -> None:
    model_path = model_assets.get_model_file("soma")
    required_assets = [model_path / "SOMA_neutral.npz", model_path / "correctives_model.pt"]
    if not all(path.exists() for path in required_assets):
        pytest.skip("SOMA assets are not available")

    expected = {"mid": 18056, "low": 4505}
    with np.load(model_path / "SOMA_neutral.npz", allow_pickle=False) as data:
        if "lod_mid_to_xlo" in data:
            expected["xlo"] = 612

    for lod, num_vertices in expected.items():
        model = SOMA(model_path=model_path, model_type="soma", rotation_type="axis_angle", lod=lod)
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
        pose = SO3.conversions.from_rotmat_to_axis_angle(rotation, xp=np)
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
    quat = SO3.conversions.from_rotmat_to_quat(rotation / scale[..., None], convention="xyzw", xp=np)
    return np.concatenate([translation, quat, scale], axis=-1)


def assert_mhr_skeleton_close(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual[:, :3], expected[:, :3], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual[:, 7:], expected[:, 7:], rtol=1e-4, atol=1e-4)
    quat_diff = np.minimum(np.abs(actual[:, 3:7] - expected[:, 3:7]), np.abs(actual[:, 3:7] + expected[:, 3:7]))
    np.testing.assert_allclose(quat_diff, 0.0, rtol=1e-4, atol=1e-4)
