import pytest

from body_models import download

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
        func(cache_dir=tmp_path / name)


@pytest.mark.fast
def test_download_smpl_finds_existing_without_credentials(tmp_path) -> None:
    cache_dir = tmp_path / "smpl"
    cache_dir.mkdir()
    for filename in download.SMPL_FILES.values():
        (cache_dir / filename).touch()

    paths = download.download_smpl(cache_dir=cache_dir)

    assert set(paths) == set(download.SMPL_FILES)
    assert all(path.exists() for path in paths.values())


@pytest.mark.fast
def test_download_smplx_finds_existing_without_credentials(tmp_path) -> None:
    cache_dir = tmp_path / "smplx"
    cache_dir.mkdir()
    for filename in download.SMPLX_FILES.values():
        (cache_dir / filename).touch()

    paths = download.download_smplx(cache_dir=cache_dir)

    assert set(paths) == set(download.SMPLX_FILES)
    assert all(path.exists() for path in paths.values())


@pytest.mark.fast
def test_download_smplh_finds_existing_without_credentials(tmp_path) -> None:
    cache_dir = tmp_path / "smplh"
    for relative_path in download.SMPLH_FILES.values():
        file_path = cache_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

    paths = download.download_smplh(cache_dir=cache_dir)

    assert set(paths) == set(download.SMPLH_FILES)
    assert all(path.exists() for path in paths.values())


@pytest.mark.fast
def test_download_mano_finds_existing_without_credentials(tmp_path) -> None:
    cache_dir = tmp_path / "mano"
    cache_dir.mkdir()
    for filename in download.MANO_FILES.values():
        (cache_dir / filename).touch()

    paths = download.download_mano(cache_dir=cache_dir)

    assert set(paths) == set(download.MANO_FILES)
    assert all(path.exists() for path in paths.values())


@pytest.mark.fast
def test_download_flame_finds_existing_without_credentials(tmp_path) -> None:
    cache_dir = tmp_path / "flame"
    cache_dir.mkdir()
    expected = cache_dir / download.FLAME_FILES[0]
    expected.touch()

    path = download.download_flame(cache_dir=cache_dir)

    assert path == expected


@pytest.mark.fast
def test_download_skel_finds_existing_without_credentials(tmp_path) -> None:
    cache_dir = tmp_path / "skel"
    cache_dir.mkdir()
    (cache_dir / "skel_male.pkl").touch()

    result = download.download_skel(cache_dir=cache_dir)

    assert result == cache_dir


@pytest.mark.fast
def test_download_skel_finds_existing_versioned_subdir(tmp_path) -> None:
    cache_dir = tmp_path / "skel"
    versioned_dir = cache_dir / "skel_models_v1.1"
    versioned_dir.mkdir(parents=True)
    (versioned_dir / "skel_male.pkl").touch()

    result = download.download_skel(cache_dir=cache_dir)

    assert result == versioned_dir


@pytest.mark.fast
def test_download_archive_raises_on_non_archive_response(tmp_path) -> None:
    source = tmp_path / "not_an_archive.txt"
    source.write_text("please log in")
    archive_path = tmp_path / "archive"

    with pytest.raises(RuntimeError, match="Check your credentials and confirm you accepted the model license"):
        download._download_archive(source.as_uri(), archive_path, "user", "pass")
