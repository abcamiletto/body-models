import numpy as np
import pytest

import model_cases
from body_models.bodies.soma.io import validate_path
import body_models.robots.brainco.io as brainco_io


@pytest.mark.parametrize(("name", "numpy_model", "_torch_model", "_jax_model", "kwargs"), model_cases.MODELS)
def test_model_loads(name, numpy_model, _torch_model, _jax_model, kwargs) -> None:
    numpy_model(**kwargs)


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
def test_brainco_get_model_path_uses_cache_without_downloading(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "brainco"
    (cache_dir / "meshes" / "left").mkdir(parents=True)
    (cache_dir / "meshes" / "right").mkdir(parents=True)
    (cache_dir / "left.xml").touch()
    (cache_dir / "right.xml").touch()

    # Override the test-suite-wide fixture that redirects config lookups to bundled
    # test assets, so this test exercises the real cache-hit path in get_model_path().
    monkeypatch.setattr(brainco_io.config, "get_model_path", lambda name: None)
    monkeypatch.setattr(brainco_io, "get_cache_dir", lambda: tmp_path)

    def _fail_download(*args, **kwargs):
        raise AssertionError("download_hf_archive should not be called on a cache hit")

    monkeypatch.setattr(brainco_io, "download_hf_archive", _fail_download)

    assert brainco_io.get_model_path() == cache_dir
