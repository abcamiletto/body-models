import zipfile

import pytest

from body_models import _cache
from body_models import _download as download

FUNCS = [
    (download.download_smpl, "smpl"),
    (download.download_smplx, "smplx"),
    (download.download_smplh, "smplh"),
    (download.download_mano, "mano"),
    (download.download_flame, "flame"),
    (download.download_skel, "skel"),
]


@pytest.mark.fast
@pytest.mark.parametrize(("func", "name"), FUNCS)
def test_missing_credentials_raises_value_error(func, name, tmp_path) -> None:
    with pytest.raises(ValueError, match="credentials"):
        func(output_dir=tmp_path / name)


@pytest.mark.fast
def test_download_smpl_finds_existing_without_credentials(tmp_path) -> None:
    cache_dir = tmp_path / "smpl"
    cache_dir.mkdir()
    for filename in download.SMPL_FILES.values():
        (cache_dir / filename).touch()

    paths = download.download_smpl(output_dir=cache_dir)

    assert set(paths) == set(download.SMPL_FILES)
    assert all(path.exists() for path in paths.values())


@pytest.mark.fast
def test_download_smplx_finds_existing_without_credentials(tmp_path) -> None:
    cache_dir = tmp_path / "smplx"
    cache_dir.mkdir()
    for filename in download.SMPLX_FILES.values():
        (cache_dir / filename).touch()

    paths = download.download_smplx(output_dir=cache_dir)

    assert set(paths) == set(download.SMPLX_FILES)
    assert all(path.exists() for path in paths.values())


@pytest.mark.fast
def test_download_smplh_finds_existing_without_credentials(tmp_path) -> None:
    cache_dir = tmp_path / "smplh"
    for relative_path in download.SMPLH_FILES.values():
        file_path = cache_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

    paths = download.download_smplh(output_dir=cache_dir)

    assert set(paths) == set(download.SMPLH_FILES)
    assert all(path.exists() for path in paths.values())


@pytest.mark.fast
def test_download_mano_finds_existing_without_credentials(tmp_path) -> None:
    cache_dir = tmp_path / "mano"
    cache_dir.mkdir()
    for filename in download.MANO_FILES.values():
        (cache_dir / filename).touch()

    paths = download.download_mano(output_dir=cache_dir)

    assert set(paths) == set(download.MANO_FILES)
    assert all(path.exists() for path in paths.values())


@pytest.mark.fast
def test_download_flame_finds_existing_without_credentials(tmp_path) -> None:
    cache_dir = tmp_path / "flame"
    cache_dir.mkdir()
    expected = cache_dir / download.FLAME_FILES[0]
    expected.touch()

    path = download.download_flame(output_dir=cache_dir)

    assert path == expected


@pytest.mark.fast
def test_download_skel_finds_existing_without_credentials(tmp_path) -> None:
    cache_dir = tmp_path / "skel"
    cache_dir.mkdir()
    (cache_dir / "skel_male.pkl").touch()

    result = download.download_skel(output_dir=cache_dir)

    assert result == cache_dir


@pytest.mark.fast
def test_download_skel_finds_existing_versioned_subdir(tmp_path) -> None:
    cache_dir = tmp_path / "skel"
    versioned_dir = cache_dir / "skel_models_v1.1"
    versioned_dir.mkdir(parents=True)
    (versioned_dir / "skel_male.pkl").touch()

    result = download.download_skel(output_dir=cache_dir)

    assert result == versioned_dir


@pytest.mark.fast
def test_download_archive_raises_on_non_archive_response(tmp_path) -> None:
    source = tmp_path / "not_an_archive.txt"
    source.write_text("please log in")
    archive_path = tmp_path / "archive"

    with pytest.raises(RuntimeError, match="Check your credentials and confirm you accepted the model license"):
        download._download_archive(source.as_uri(), archive_path, "user", "pass")


@pytest.mark.fast
def test_failed_download_preserves_existing_assets(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "smpl"
    output_dir.mkdir()
    existing = output_dir / "existing.pkl"
    existing.write_text("model")

    def fail(*args, **kwargs):
        raise RuntimeError("network failure")

    monkeypatch.setattr(download, "_download_archive", fail)

    with pytest.raises(RuntimeError, match="network failure"):
        download._fetch("SMPL", "https://example.invalid", output_dir, "user", "pass")

    assert existing.read_text() == "model"


@pytest.mark.fast
def test_archive_replaces_destination_only_after_extraction(tmp_path) -> None:
    archive_path = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("new/model.npz", "model")

    output_dir = tmp_path / "model"
    output_dir.mkdir()
    (output_dir / "old.npz").touch()

    _cache.extract_archive(archive_path, output_dir)

    assert (output_dir / "new" / "model.npz").read_text() == "model"
    assert not (output_dir / "old.npz").exists()
