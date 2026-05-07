from body_models.flame import io
from model_io_helpers import assert_same_weights, model_data, write_model_files


def test_load_model_data_returns_same_weights_for_pkl_and_npz(tmp_path) -> None:
    pkl_path, npz_path = write_model_files(
        tmp_path,
        "FLAME_NEUTRAL",
        model_data(num_joints=5, num_shapes=400),
    )

    assert_same_weights(io.load_model_data(pkl_path), io.load_model_data(npz_path))
