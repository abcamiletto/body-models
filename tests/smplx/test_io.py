from body_models.smplx import io
from model_io_helpers import assert_same_weights, smplx_model_data, write_model_files


def test_load_model_data_returns_same_weights_for_pkl_and_npz(tmp_path) -> None:
    pkl_path, npz_path = write_model_files(tmp_path, "SMPLX_NEUTRAL", smplx_model_data())

    assert_same_weights(io.load_model_data(pkl_path), io.load_model_data(npz_path))


def test_validate_path_accepts_pkl(tmp_path) -> None:
    path = tmp_path / "SMPLX_NEUTRAL.pkl"
    path.write_bytes(b"")

    assert io.validate_path(path) == path
