import pickle

import numpy as np

from body_models.smpl.io import load_model_data


def test_load_model_data_returns_same_weights_for_pkl_and_npz(tmp_path) -> None:
    model_data = {
        "v_template": np.arange(6, dtype=np.float32).reshape(2, 3),
        "f": np.array([[0, 1, 0]], dtype=np.int32),
        "weights": np.arange(48, dtype=np.float32).reshape(2, 24),
        "shapedirs": np.arange(60, dtype=np.float32).reshape(2, 3, 10),
        "posedirs": np.arange(1242, dtype=np.float32).reshape(2, 3, 207),
        "J_regressor": np.eye(2, dtype=np.float32),
        "kintree_table": np.vstack(
            [
                np.array([0, 0], dtype=np.int64),
                np.array([0, 1], dtype=np.int64),
            ]
        ),
    }
    pkl_path = tmp_path / "SMPL_NEUTRAL.pkl"
    npz_path = tmp_path / "SMPL_NEUTRAL.npz"

    with open(pkl_path, "wb") as f:
        pickle.dump(model_data, f, protocol=2)
    np.savez(npz_path, **model_data)

    pkl_weights = load_model_data(pkl_path)
    npz_weights = load_model_data(npz_path)

    np.testing.assert_array_equal(pkl_weights.v_template, npz_weights.v_template)
    np.testing.assert_array_equal(pkl_weights.faces, npz_weights.faces)
    np.testing.assert_array_equal(pkl_weights.lbs_weights, npz_weights.lbs_weights)
    np.testing.assert_array_equal(pkl_weights.shapedirs, npz_weights.shapedirs)
    np.testing.assert_array_equal(pkl_weights.posedirs, npz_weights.posedirs)
    np.testing.assert_array_equal(pkl_weights.j_template, npz_weights.j_template)
    np.testing.assert_array_equal(pkl_weights.j_shapedirs, npz_weights.j_shapedirs)
    assert pkl_weights.parents == npz_weights.parents
    assert pkl_weights.kinematic_fronts == npz_weights.kinematic_fronts
