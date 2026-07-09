from __future__ import annotations

import shutil
import tarfile
import urllib.parse
import urllib.request
from urllib.error import HTTPError
import zipfile
from pathlib import Path

from .cache import get_cache_dir

SMPL_URL = "https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip"
SMPLX_URL = "https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip"
SMPLH_URL = "https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=smplh.tar.xz"
MANO_URL = "https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=mano_v1_2.zip"
SKEL_URL = "https://download.is.tue.mpg.de/download.php?domain=skel&sfile=skel_models_v1.1.zip&resume=1"
FLAME_URL = "https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2023.zip&resume=1"

SMPL_FILES = {
    "smpl-neutral": "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
    "smpl-female": "basicmodel_f_lbs_10_207_0_v1.1.0.pkl",
    "smpl-male": "basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
}
SMPLX_FILES = {
    "smplx-neutral": "SMPLX_NEUTRAL.npz",
    "smplx-female": "SMPLX_FEMALE.npz",
    "smplx-male": "SMPLX_MALE.npz",
}
SMPLH_FILES = {
    "smplh-neutral": "neutral/model.npz",
    "smplh-female": "female/model.npz",
    "smplh-male": "male/model.npz",
}
MANO_FILES = {
    "mano-right": "MANO_RIGHT.pkl",
    "mano-left": "MANO_LEFT.pkl",
}
FLAME_FILES = ["flame2023.pkl", "FLAME_NEUTRAL.pkl", "generic_model.pkl", "flame2023_no_jaw.pkl"]


def _download_archive(url: str, archive_path: Path, username: str, password: str) -> None:
    """POST MPI credentials, stream the archive, and fail loudly on a non-archive response."""
    post_data = urllib.parse.urlencode({"username": username, "password": password}).encode()
    request = urllib.request.Request(url, data=post_data)
    try:
        with urllib.request.urlopen(request) as response, archive_path.open("wb") as f:
            shutil.copyfileobj(response, f)
    except HTTPError as exc:
        snippet = exc.read(200).decode(errors="ignore").strip()
        raise RuntimeError(
            f"Download failed with HTTP {exc.code}. Check your credentials and confirm you accepted the model license."
            + (f" Response started with: {snippet!r}" if snippet else "")
        ) from exc
    if zipfile.is_zipfile(archive_path) or tarfile.is_tarfile(archive_path):
        return
    snippet = archive_path.read_text(errors="ignore")[:200].strip()
    raise RuntimeError(
        "Download failed. Check your credentials and confirm you accepted the model license."
        + (f" Response started with: {snippet!r}" if snippet else "")
    )


def _extract_archive(archive_path: Path, dest: Path) -> None:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
        return
    with tarfile.open(archive_path) as tf:
        _extract_tar(tf, dest)


def _fetch(name: str, url: str, cache_dir: Path, username: str | None, password: str | None) -> None:
    if username is None or password is None:
        raise ValueError(f"{name} credentials are required to download the model.")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "archive"
    _download_archive(url, archive_path, username, password)
    _extract_archive(archive_path, cache_dir)
    archive_path.unlink(missing_ok=True)


def download_smpl(
    cache_dir: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Path]:
    cache_dir = Path(cache_dir) if cache_dir else get_cache_dir() / "smpl"
    paths = {model: next(cache_dir.rglob(name), None) for model, name in SMPL_FILES.items()}
    if None not in paths.values():
        return {model: path for model, path in paths.items() if path is not None}

    _fetch("SMPL", SMPL_URL, cache_dir, username, password)

    paths = {model: next(cache_dir.rglob(name), None) for model, name in SMPL_FILES.items()}
    if None in paths.values():
        raise FileNotFoundError(f"Expected SMPL model files were not found in {cache_dir}")

    return {model: path for model, path in paths.items() if path is not None}


def download_smplx(
    cache_dir: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Path]:
    cache_dir = Path(cache_dir) if cache_dir else get_cache_dir() / "smplx"
    paths = {model: next(cache_dir.rglob(name), None) for model, name in SMPLX_FILES.items()}
    if None not in paths.values():
        return {model: path for model, path in paths.items() if path is not None}

    _fetch("SMPL-X", SMPLX_URL, cache_dir, username, password)

    paths = {model: next(cache_dir.rglob(name), None) for model, name in SMPLX_FILES.items()}
    if None in paths.values():
        raise FileNotFoundError(f"Expected SMPL-X model files were not found in {cache_dir}")

    return {model: path for model, path in paths.items() if path is not None}


def download_smplh(
    cache_dir: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Path]:
    cache_dir = Path(cache_dir) if cache_dir else get_cache_dir() / "smplh"
    paths = {model: _find_relative_path(cache_dir, name) for model, name in SMPLH_FILES.items()}
    if None not in paths.values():
        return {model: path for model, path in paths.items() if path is not None}

    _fetch("SMPL-H", SMPLH_URL, cache_dir, username, password)

    paths = {model: _find_relative_path(cache_dir, name) for model, name in SMPLH_FILES.items()}
    if None in paths.values():
        raise FileNotFoundError(f"Expected SMPL-H model files were not found in {cache_dir}")

    return {model: path for model, path in paths.items() if path is not None}


def download_mano(
    cache_dir: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Path]:
    cache_dir = Path(cache_dir) if cache_dir else get_cache_dir() / "mano"
    paths = {model: next(cache_dir.rglob(name), None) for model, name in MANO_FILES.items()}
    if None not in paths.values():
        return {model: path for model, path in paths.items() if path is not None}

    _fetch("MANO", MANO_URL, cache_dir, username, password)

    paths = {model: next(cache_dir.rglob(name), None) for model, name in MANO_FILES.items()}
    if None in paths.values():
        raise FileNotFoundError(f"Expected MANO model files were not found in {cache_dir}")

    return {model: path for model, path in paths.items() if path is not None}


def download_flame(
    cache_dir: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> Path:
    cache_dir = Path(cache_dir) if cache_dir else get_cache_dir() / "flame"
    for name in FLAME_FILES:
        path = next(cache_dir.rglob(name), None)
        if path is not None:
            return path

    _fetch("FLAME", FLAME_URL, cache_dir, username, password)

    for name in FLAME_FILES:
        path = next(cache_dir.rglob(name), None)
        if path is not None:
            return path

    raise FileNotFoundError(f"Expected FLAME model file was not found in {cache_dir}")


def download_skel(
    cache_dir: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> Path:
    cache_dir = Path(cache_dir) if cache_dir else get_cache_dir() / "skel"
    if (cache_dir / "skel_male.pkl").exists():
        return cache_dir
    existing_dir = next(cache_dir.glob("skel_models_v*"), None)
    if existing_dir is not None and (existing_dir / "skel_male.pkl").exists():
        return existing_dir

    _fetch("SKEL", SKEL_URL, cache_dir, username, password)

    if (cache_dir / "skel_male.pkl").exists():
        return cache_dir
    existing_dir = next(cache_dir.glob("skel_models_v*"), None)
    if existing_dir is not None and (existing_dir / "skel_male.pkl").exists():
        return existing_dir

    raise FileNotFoundError(f"Expected SKEL model files were not found in {cache_dir}")


def _find_relative_path(cache_dir: Path, relative_path: str) -> Path | None:
    wanted = Path(relative_path)
    for path in cache_dir.rglob(wanted.name):
        if len(path.parts) >= len(wanted.parts) and path.parts[-len(wanted.parts) :] == wanted.parts:
            return path
    return None


def _extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    members = tf.getmembers()
    for member in members:
        member_path = Path(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    try:
        tf.extractall(dest, members=members, filter="data")
    except TypeError:
        tf.extractall(dest, members=members)
