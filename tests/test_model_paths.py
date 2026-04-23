from pathlib import Path

import pytest

from body_models import config
from body_models.flame.io import get_model_path as get_flame_model_path
from body_models.mhr.io import get_model_path as get_mhr_model_path
from body_models.soma.io import SOMA_CORE_ASSET, get_identity_model_path, get_model_path as get_soma_model_path


def test_flame_get_model_path_rejects_directories(tmp_path: Path) -> None:
    model_dir = tmp_path / "flame"
    model_dir.mkdir()
    (model_dir / "FLAME_NEUTRAL.pkl").write_bytes(b"stub")

    with pytest.raises(ValueError, match="Directory paths are no longer supported"):
        get_flame_model_path(model_dir)


def test_mhr_get_model_path_rejects_checkpoint_files(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mhr_model.pt"
    checkpoint.write_bytes(b"stub")

    with pytest.raises(ValueError, match="Expected an MHR model directory"):
        get_mhr_model_path(checkpoint)


def test_soma_get_model_path_rejects_core_asset_files(tmp_path: Path) -> None:
    checkpoint = tmp_path / SOMA_CORE_ASSET
    checkpoint.write_bytes(b"stub")

    with pytest.raises(ValueError, match="Expected a SOMA asset directory"):
        get_soma_model_path(checkpoint)


@pytest.mark.parametrize(
    ("model_type", "config_key", "filename"),
    [
        ("smpl", "smpl-neutral", "SMPL_NEUTRAL.pkl"),
        ("smplx", "smplx-neutral", "SMPLX_NEUTRAL.npz"),
    ],
)
def test_soma_identity_model_path_rejects_directory_configs(
    model_type: str,
    config_key: str,
    filename: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / model_type
    model_dir.mkdir()
    (model_dir / filename).write_bytes(b"stub")

    monkeypatch.setattr(
        config,
        "get_model_path",
        lambda model: model_dir if model == config_key else None,
    )

    with pytest.raises(ValueError, match="Directory paths are no longer supported"):
        get_identity_model_path(model_type)
