from pathlib import Path

import pytest

from body_models import config


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _valid_model_paths(tmp_path: Path) -> dict[str, Path]:
    skel_dir = tmp_path / "skel"
    _touch(skel_dir / "skel_male.pkl")
    _touch(skel_dir / "skel_female.pkl")

    anny_dir = tmp_path / "anny"
    (anny_dir / "data" / "mpfb2").mkdir(parents=True)

    mhr_dir = tmp_path / "mhr"
    _touch(mhr_dir / "mhr_model.pt")

    g1_dir = tmp_path / "g1"
    _touch(g1_dir / "xml" / "g1.xml")
    (g1_dir / "meshes" / "g1").mkdir(parents=True)

    soma_dir = tmp_path / "soma"
    _touch(soma_dir / "SOMA_neutral.npz")
    _touch(soma_dir / "correctives_model.pt")

    garment_dir = tmp_path / "garment"
    _touch(garment_dir / "garment_measurements.npz")

    return {
        "smpl-neutral": _touch(tmp_path / "SMPL_NEUTRAL.pkl"),
        "smpl-male": _touch(tmp_path / "SMPL_MALE.pkl"),
        "smpl-female": _touch(tmp_path / "SMPL_FEMALE.npz"),
        "smplx-neutral": _touch(tmp_path / "SMPLX_NEUTRAL.npz"),
        "smplx-male": _touch(tmp_path / "SMPLX_MALE.npz"),
        "smplx-female": _touch(tmp_path / "SMPLX_FEMALE.npz"),
        "smplh-neutral": _touch(tmp_path / "smplh" / "neutral" / "model.npz"),
        "smplh-male": _touch(tmp_path / "smplh" / "male" / "model.npz"),
        "smplh-female": _touch(tmp_path / "smplh" / "female" / "model.npz"),
        "mano-right": _touch(tmp_path / "mano" / "MANO_RIGHT.pkl"),
        "mano-left": _touch(tmp_path / "mano" / "MANO_LEFT.pkl"),
        "skel": skel_dir,
        "anny": anny_dir,
        "mhr": mhr_dir,
        "flame": _touch(tmp_path / "FLAME_NEUTRAL.pkl"),
        "g1": g1_dir,
        "soma": soma_dir,
        "garment-measurements": garment_dir,
    }


def test_set_model_path_validates_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config" / "config.toml")

    paths = _valid_model_paths(tmp_path)
    for model, path in paths.items():
        config.set_model_path(model, str(path))

    stored_paths = config.get_config()["paths"]
    assert stored_paths == {model: str(path) for model, path in sorted(paths.items())}

    with pytest.raises(ValueError, match="Expected an SMPL model file"):
        config.set_model_path("smpl-neutral", str(tmp_path))

    with pytest.raises(ValueError, match="Expected an MHR model directory"):
        config.set_model_path("mhr", str(_touch(tmp_path / "mhr_model.pt")))


def test_gendered_file_models_use_gender_only_for_config_lookup(tmp_path: Path) -> None:
    from body_models.smpl.io import get_model_path as get_smpl_model_path
    from body_models.smplh.io import get_model_path as get_smplh_model_path
    from body_models.smplx.io import get_model_path as get_smplx_model_path
    from body_models.mano.io import get_model_path as get_mano_model_path

    smpl_path = _touch(tmp_path / "SMPL_NEUTRAL.pkl")
    smplx_path = _touch(tmp_path / "SMPLX_NEUTRAL.npz")
    smplh_path = _touch(tmp_path / "SMPLH_NEUTRAL.npz")
    mano_path = _touch(tmp_path / "MANO_RIGHT.pkl")

    assert get_smpl_model_path(smpl_path, gender=None) == smpl_path
    assert get_smplx_model_path(smplx_path, gender=None) == smplx_path
    assert get_smplh_model_path(smplh_path, gender=None) == smplh_path
    assert get_mano_model_path(mano_path, side=None) == mano_path

    with pytest.raises(ValueError, match="gender is only supported"):
        get_smpl_model_path(smpl_path, gender="neutral")

    with pytest.raises(ValueError, match="gender is only supported"):
        get_smplx_model_path(smplx_path, gender="neutral")

    with pytest.raises(ValueError, match="gender is only supported"):
        get_smplh_model_path(smplh_path, gender="neutral")

    with pytest.raises(ValueError, match="side is only supported"):
        get_mano_model_path(mano_path, side="right")
