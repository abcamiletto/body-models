import numpy as np
import pytest

from body_models.mhr import _io as mhr_io
from body_models.soma._io import validate_path


def test_soma_slim_npz_asset_layout_requires_rig_fields(tmp_path) -> None:
    np.savez(tmp_path / "SOMA_neutral.npz", mean=np.zeros((1, 3), dtype=np.float32))
    (tmp_path / "correctives_model.pt").touch()

    with pytest.raises(FileNotFoundError, match="missing required NPZ fields"):
        validate_path(tmp_path)


def test_soma_upstream_021_asset_layout_requires_preprocessing(tmp_path) -> None:
    np.savez(tmp_path / "SOMA_neutral.npz", mean=np.zeros((1, 3), dtype=np.float32))
    (tmp_path / "correctives_model.pt").touch()
    (tmp_path / "SOMA_template_rig.usda").touch()
    (tmp_path / "SOMA_procedural_transforms.json").touch()

    with pytest.raises(FileNotFoundError, match="body-models preprocess-soma"):
        validate_path(tmp_path)


@pytest.mark.fast
def test_mhr_has_model_requires_all_hosted_lods(tmp_path) -> None:
    for name in mhr_io.MHR_ASSETS[:-1]:
        (tmp_path / name).touch()

    assert not mhr_io._has_model(tmp_path)

    (tmp_path / mhr_io.MHR_ASSETS[-1]).touch()

    assert mhr_io._has_model(tmp_path)


@pytest.mark.fast
def test_mhr_validation_reports_missing_default_assets(tmp_path) -> None:
    (tmp_path / "mhr_model.pt").touch()

    with pytest.raises(
        FileNotFoundError,
        match=r"corrective_activation\.npz, corrective_blendshapes_lod1\.npz",
    ):
        mhr_io.validate_path(tmp_path)


@pytest.mark.fast
def test_mhr_loading_reports_missing_selected_lod_assets(tmp_path) -> None:
    for name in mhr_io._MHR_DEFAULT_ASSETS:
        (tmp_path / name).touch()

    with pytest.raises(
        FileNotFoundError,
        match=r"corrective_blendshapes_lod2\.npz, mhr_lod2\.npz",
    ):
        mhr_io.load_model_data(tmp_path, lod=2)
