from pathlib import Path
import zipfile

from body_models import fetch


def _make_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def test_download_smpl_uses_downloaded_archive_and_reuses_cache(tmp_path, monkeypatch) -> None:
    source_zip = tmp_path / "smpl-source.zip"
    _make_zip(
        source_zip,
        {
            "SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl": b"neutral",
            "SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl": b"female",
            "SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl": b"male",
        },
    )

    def fake_download(url: str, archive_path: Path, username: str, password: str) -> None:
        assert url == fetch.SMPL_URL
        assert username == "user"
        assert password == "secret"
        archive_path.write_bytes(source_zip.read_bytes())

    monkeypatch.setattr(fetch, "_download_zip", fake_download)

    cache_dir = tmp_path / "smpl-cache"
    paths = fetch.download_smpl(cache_dir=cache_dir, username="user", password="secret")
    assert {key: path.read_bytes() for key, path in paths.items()} == {
        "smpl-female": b"female",
        "smpl-male": b"male",
        "smpl-neutral": b"neutral",
    }

    paths = fetch.download_smpl(cache_dir=cache_dir)
    assert {key: path.read_bytes() for key, path in paths.items()} == {
        "smpl-female": b"female",
        "smpl-male": b"male",
        "smpl-neutral": b"neutral",
    }


def test_download_smplx_uses_downloaded_archive_and_reuses_cache(tmp_path, monkeypatch) -> None:
    source_zip = tmp_path / "smplx-source.zip"
    _make_zip(
        source_zip,
        {
            "models/smplx/SMPLX_NEUTRAL.npz": b"neutral",
            "models/smplx/SMPLX_FEMALE.npz": b"female",
            "models/smplx/SMPLX_MALE.npz": b"male",
        },
    )

    def fake_download(url: str, archive_path: Path, username: str, password: str) -> None:
        assert url == fetch.SMPLX_URL
        assert username == "user"
        assert password == "secret"
        archive_path.write_bytes(source_zip.read_bytes())

    monkeypatch.setattr(fetch, "_download_zip", fake_download)

    cache_dir = tmp_path / "smplx-cache"
    paths = fetch.download_smplx(cache_dir=cache_dir, username="user", password="secret")
    assert {key: path.read_bytes() for key, path in paths.items()} == {
        "smplx-female": b"female",
        "smplx-male": b"male",
        "smplx-neutral": b"neutral",
    }

    paths = fetch.download_smplx(cache_dir=cache_dir)
    assert {key: path.read_bytes() for key, path in paths.items()} == {
        "smplx-female": b"female",
        "smplx-male": b"male",
        "smplx-neutral": b"neutral",
    }


def test_download_flame_uses_downloaded_archive_and_reuses_cache(tmp_path, monkeypatch) -> None:
    source_zip = tmp_path / "flame-source.zip"
    _make_zip(source_zip, {"FLAME2023/flame2023.pkl": b"flame"})

    def fake_download(url: str, archive_path: Path, username: str, password: str) -> None:
        assert url == fetch.FLAME_URL
        assert username == "user"
        assert password == "secret"
        archive_path.write_bytes(source_zip.read_bytes())

    monkeypatch.setattr(fetch, "_download_zip", fake_download)

    cache_dir = tmp_path / "flame-cache"
    path = fetch.download_flame(cache_dir=cache_dir, username="user", password="secret")
    assert path.read_bytes() == b"flame"

    path = fetch.download_flame(cache_dir=cache_dir)
    assert path.read_bytes() == b"flame"


def test_download_skel_uses_downloaded_archive_and_reuses_cache(tmp_path, monkeypatch) -> None:
    source_zip = tmp_path / "skel-source.zip"
    _make_zip(
        source_zip,
        {
            "skel_models_v1.1/skel_male.pkl": b"male",
            "skel_models_v1.1/skel_female.pkl": b"female",
        },
    )

    def fake_download(url: str, archive_path: Path, username: str, password: str) -> None:
        assert url == fetch.SKEL_URL
        assert username == "user"
        assert password == "secret"
        archive_path.write_bytes(source_zip.read_bytes())

    monkeypatch.setattr(fetch, "_download_zip", fake_download)

    cache_dir = tmp_path / "skel-cache"
    path = fetch.download_skel(cache_dir=cache_dir, username="user", password="secret")
    assert path.name == "skel_models_v1.1"
    assert (path / "skel_male.pkl").read_bytes() == b"male"
    assert (path / "skel_female.pkl").read_bytes() == b"female"

    path = fetch.download_skel(cache_dir=cache_dir)
    assert path.name == "skel_models_v1.1"
