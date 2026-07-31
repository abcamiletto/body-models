import json
import struct
from pathlib import Path

import model_assets
import numpy as np
import pytest

from body_models.smpl_humanoid import SmplxMannequin, generate
from body_models.smplx import SMPLX

SOURCE = Path("src/body_models/smpl_humanoid/assets/smpl_robot_professional.glb")


def test_bundled_character_is_rigid_and_complete() -> None:
    parts = generate._load_source_geometries(SOURCE)
    assert {part.joint_index for part in parts} == set(range(54))

    with SOURCE.open("rb") as handle:
        magic, version, _ = struct.unpack("<4sII", handle.read(12))
        chunk_length, chunk_type = struct.unpack("<II", handle.read(8))
        document = json.loads(handle.read(chunk_length))

    assert (magic, version, chunk_type) == (b"glTF", 2, 0x4E4F534A)
    assert "skins" not in document
    assert "animations" not in document


def test_smplx_mannequin_shape_moves_joints_and_rigid_meshes() -> None:
    model_path = model_assets.get_test_model_file("smplx")
    if not model_path.is_file():
        pytest.skip("SMPL-X test asset is unavailable")

    mannequin = SmplxMannequin(
        smplx_model=SMPLX(model_path=model_path, flat_hand_mean=True),
    )
    shape = np.zeros(10, dtype=np.float32)
    shape[0] = 0.8
    identity = mannequin.prepare_identity(shape)
    params = mannequin.get_rest_pose()
    params.pop("shape")
    params.pop("expression")
    vertices = mannequin.forward_vertices(**params, identity=identity)
    skeleton = mannequin.forward_skeleton(**params, identity=identity)

    np.testing.assert_allclose(vertices, identity["rest_vertices"], atol=1e-6)
    np.testing.assert_allclose(skeleton[:, :3, 3], identity["rest_joints"], atol=1e-6)

    by_name = {name: index for index, name in enumerate(mannequin.joint_names)}
    reflection = np.array((-1.0, 1.0, 1.0), dtype=np.float32)
    left_wrist = identity["rest_joints"][by_name["L_Wrist"]]
    right_wrist = identity["rest_joints"][by_name["R_Wrist"]]
    np.testing.assert_allclose(left_wrist * reflection, right_wrist, atol=1e-6)
