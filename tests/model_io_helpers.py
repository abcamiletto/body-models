import pickle
from dataclasses import fields
from pathlib import Path

import numpy as np


def write_model_files(tmp_path: Path, name: str, model_data: dict) -> tuple[Path, Path]:
    pkl_path = tmp_path / f"{name}.pkl"
    npz_path = tmp_path / f"{name}.npz"

    with open(pkl_path, "wb") as f:
        pickle.dump(model_data, f, protocol=2)
    np.savez(npz_path, **model_data)

    return pkl_path, npz_path


def assert_same_weights(left, right) -> None:
    for field in fields(left):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, np.ndarray):
            np.testing.assert_array_equal(left_value, right_value)
        else:
            assert left_value == right_value


def model_data(num_joints: int, num_shapes: int) -> dict:
    num_vertices = 3
    return {
        "v_template": np.arange(num_vertices * 3, dtype=np.float32).reshape(num_vertices, 3),
        "f": np.array([[0, 1, 2]], dtype=np.int32),
        "weights": np.arange(num_vertices * num_joints, dtype=np.float32).reshape(num_vertices, num_joints),
        "shapedirs": np.arange(num_vertices * 3 * num_shapes, dtype=np.float32).reshape(num_vertices, 3, num_shapes),
        "posedirs": np.arange(num_vertices * 3 * (num_joints - 1) * 9, dtype=np.float32).reshape(
            num_vertices,
            3,
            (num_joints - 1) * 9,
        ),
        "J_regressor": np.arange(num_joints * num_vertices, dtype=np.float32).reshape(num_joints, num_vertices),
        "kintree_table": kintree_table(num_joints),
    }


def mano_model_data() -> dict:
    data = model_data(num_joints=16, num_shapes=10)
    data["hands_mean"] = np.arange(45, dtype=np.float32)
    return data


def smplh_model_data() -> dict:
    data = model_data(num_joints=52, num_shapes=16)
    data["hands_meanl"] = np.arange(45, dtype=np.float32)
    data["hands_meanr"] = np.arange(45, dtype=np.float32) + 45
    return data


def smplx_model_data() -> dict:
    data = model_data(num_joints=55, num_shapes=400)
    data["hands_meanl"] = np.arange(45, dtype=np.float32)
    data["hands_meanr"] = np.arange(45, dtype=np.float32) + 45
    data["joint2num"] = {f"joint_{idx}": idx for idx in range(55)}
    return data


def kintree_table(num_joints: int) -> np.ndarray:
    parents = np.arange(num_joints, dtype=np.int64) - 1
    parents[0] = 0
    return np.vstack([parents, np.arange(num_joints, dtype=np.int64)])
