import model_assets
import numpy as np
import pytest

from body_models import _config as config
from body_models.mano import _io as mano_io
from body_models.mano.numpy import MANO
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
def test_mhr_download_requires_all_hosted_lods(tmp_path) -> None:
    for name in mhr_io.MHR_ASSETS[:-1]:
        (tmp_path / name).touch()

    with pytest.raises(FileNotFoundError, match=mhr_io.MHR_ASSETS[-1]):
        mhr_io._validate_hosted(tmp_path)

    (tmp_path / mhr_io.MHR_ASSETS[-1]).touch()

    assert mhr_io._validate_hosted(tmp_path) == tmp_path


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


def test_mano_flip_shapedirs_mirrors_shape_space() -> None:
    path = model_assets.get_model_file("mano")
    official = mano_io.load_model_data(path)
    fixed = mano_io.load_model_data(path, flip_shapedirs=True)

    np.testing.assert_array_equal(fixed.shapedirs[:, 0], -official.shapedirs[:, 0])
    np.testing.assert_array_equal(fixed.shapedirs[:, 1:], official.shapedirs[:, 1:])
    np.testing.assert_allclose(fixed.j_shapedirs[:, 0], -official.j_shapedirs[:, 0], atol=1e-6)
    np.testing.assert_allclose(fixed.j_shapedirs[:, 1:], official.j_shapedirs[:, 1:], atol=1e-6)


def test_mano_flip_shapedirs_requires_left_side(monkeypatch) -> None:
    path = model_assets.get_model_file("mano")
    monkeypatch.setattr(config, "get_model_path", lambda key: path)

    assert MANO(side="left", flip_shapedirs=True).side == "left"
    with pytest.raises(ValueError, match="flip_shapedirs requires side='left'"):
        MANO(side="right", flip_shapedirs=True)
    with pytest.raises(ValueError, match="flip_shapedirs requires side='left'"):
        MANO(model_path=path, flip_shapedirs=True)


@pytest.mark.fast
def test_unconfigured_licensed_asset_names_the_cli_commands(monkeypatch) -> None:
    monkeypatch.setattr(config, "get_model_path", lambda key: None)

    with pytest.raises(FileNotFoundError, match=r"body-models download mano.*body-models set mano-left"):
        config.resolve_model_path("mano", variant="left")


@pytest.mark.fast
def test_unconfigured_public_asset_downloads(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "get_model_path", lambda key: None)
    monkeypatch.setattr(mhr_io, "download_model", lambda: tmp_path)

    assert config.resolve_model_path("mhr") == tmp_path


@pytest.mark.fast
def test_explicit_path_wins_over_config_and_is_validated(tmp_path, monkeypatch) -> None:
    configured = tmp_path / "configured.pkl"
    explicit = tmp_path / "explicit.npz"
    wrong_suffix = tmp_path / "model.txt"
    for path in (configured, explicit, wrong_suffix):
        path.touch()
    monkeypatch.setattr(config, "get_model_path", lambda key: configured)

    assert config.resolve_model_path("smpl", explicit) == explicit
    assert config.resolve_model_path("smpl", variant="male") == configured
    with pytest.raises(ValueError, match=r"Expected a SMPL \.pkl or \.npz file"):
        config.resolve_model_path("smpl", wrong_suffix)
    with pytest.raises(ValueError, match="got directory"):
        config.validate_model_path("smpl-neutral", tmp_path)


@pytest.mark.fast
def test_variant_models_require_exactly_one_of_path_or_variant(tmp_path) -> None:
    with pytest.raises(ValueError, match="exactly one of model_path or side"):
        MANO()
    with pytest.raises(ValueError, match="exactly one of model_path or side"):
        MANO(model_path=tmp_path / "model.pkl", side="left")
