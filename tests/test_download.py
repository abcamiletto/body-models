import zipfile

import numpy as np
import pytest

from body_models import _cache
from body_models import _download as download
from body_models._catalog import FAMILIES

LICENSED = [name for name, family in FAMILIES.items() if family.licensed is not None]


@pytest.mark.fast
@pytest.mark.parametrize("name", LICENSED)
def test_missing_credentials_raises_value_error(name, tmp_path) -> None:
    with pytest.raises(ValueError, match="credentials"):
        download.download_licensed(name, output_dir=tmp_path / name)


@pytest.mark.fast
@pytest.mark.parametrize("name", LICENSED)
def test_download_licensed_finds_existing_without_credentials(name, tmp_path) -> None:
    files = FAMILIES[name].licensed.files
    for relative_path in files.values():
        file_path = tmp_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

    paths = download.download_licensed(name, output_dir=tmp_path)

    assert paths == {key: tmp_path / relative_path for key, relative_path in files.items()}


@pytest.mark.fast
def test_download_licensed_finds_existing_versioned_subdir(tmp_path) -> None:
    versioned_dir = tmp_path / "skel_models_v1.1"
    versioned_dir.mkdir()
    (versioned_dir / "skel_male.pkl").touch()
    (versioned_dir / "skel_female.pkl").touch()

    paths = download.download_licensed("skel", output_dir=tmp_path)

    assert paths == {"skel-male": versioned_dir / "skel_male.pkl", "skel-female": versioned_dir / "skel_female.pkl"}


@pytest.mark.fast
def test_download_licensed_rejects_ambiguous_files(tmp_path) -> None:
    for subdir in ("a", "b"):
        (tmp_path / subdir).mkdir()
        (tmp_path / subdir / "flame2023.pkl").touch()

    with pytest.raises(ValueError, match="Ambiguous flame asset"):
        download.download_licensed("flame", output_dir=tmp_path)


@pytest.mark.fast
def test_download_public_assets_skips_present_assets(tmp_path, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("present assets must not be downloaded")

    monkeypatch.setattr(download, "hf_hub_download", fail)

    assert download.download_public_assets("pkg", tmp_path, lambda path: path) == tmp_path


@pytest.mark.fast
def test_download_public_assets_fetches_until_valid(tmp_path, monkeypatch) -> None:
    archive_path = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("model.npz", "model")
    requested = []

    def hf_hub_download(repo_id, filename, cache_dir):
        requested.append(filename)
        return str(archive_path)

    def validate(path):
        if not (path / "model.npz").is_file():
            raise FileNotFoundError(path)
        return path / "model.npz"

    monkeypatch.setattr(download, "hf_hub_download", hf_hub_download)

    output_dir = tmp_path / "pkg"
    assert download.download_public_assets("pkg", output_dir, validate) == output_dir / "model.npz"
    assert requested == ["pkg/assets.zip"]


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


@pytest.mark.fast
def test_failed_npz_write_preserves_existing_cache(tmp_path, monkeypatch) -> None:
    cache_file = tmp_path / "cache.npz"
    _cache.write_npz_atomic(cache_file, values=np.array([1]))
    original = cache_file.read_bytes()

    def fail(path, **arrays):
        path.write_bytes(b"partial")
        raise RuntimeError("write failed")

    monkeypatch.setattr(_cache.np, "savez_compressed", fail)

    with pytest.raises(RuntimeError, match="write failed"):
        _cache.write_npz_atomic(cache_file, values=np.array([2]))

    assert cache_file.read_bytes() == original
    assert list(tmp_path.iterdir()) == [cache_file]
