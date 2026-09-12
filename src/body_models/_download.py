from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import HTTPError

from ._cache import extract_archive, get_cache_dir

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


def _fetch(name: str, url: str, output_dir: Path, username: str | None, password: str | None) -> None:
    if username is None or password is None:
        raise ValueError(f"{name} credentials are required to download the model.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        archive_path = Path(temporary) / "archive"
        _download_archive(url, archive_path, username, password)
        extract_archive(archive_path, output_dir)


def download_smpl(
    output_dir: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir) if output_dir else get_cache_dir() / "smpl"
    return _download_files("SMPL", SMPL_URL, SMPL_FILES, output_dir, username, password)


def download_smplx(
    output_dir: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir) if output_dir else get_cache_dir() / "smplx"
    return _download_files("SMPL-X", SMPLX_URL, SMPLX_FILES, output_dir, username, password)


def download_smplh(
    output_dir: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir) if output_dir else get_cache_dir() / "smplh"
    return _download_files("SMPL-H", SMPLH_URL, SMPLH_FILES, output_dir, username, password)


def download_mano(
    output_dir: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir) if output_dir else get_cache_dir() / "mano"
    return _download_files("MANO", MANO_URL, MANO_FILES, output_dir, username, password)


def download_flame(
    output_dir: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> Path:
    output_dir = Path(output_dir) if output_dir else get_cache_dir() / "flame"
    for name in FLAME_FILES:
        path = next(output_dir.rglob(name), None)
        if path is not None:
            return path

    _fetch("FLAME", FLAME_URL, output_dir, username, password)

    for name in FLAME_FILES:
        path = next(output_dir.rglob(name), None)
        if path is not None:
            return path

    raise FileNotFoundError(f"Expected FLAME model file was not found in {output_dir}")


def download_skel(
    output_dir: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> Path:
    output_dir = Path(output_dir) if output_dir else get_cache_dir() / "skel"
    if (output_dir / "skel_male.pkl").exists():
        return output_dir
    existing_dir = next(output_dir.glob("skel_models_v*"), None)
    if existing_dir is not None and (existing_dir / "skel_male.pkl").exists():
        return existing_dir

    _fetch("SKEL", SKEL_URL, output_dir, username, password)

    if (output_dir / "skel_male.pkl").exists():
        return output_dir
    existing_dir = next(output_dir.glob("skel_models_v*"), None)
    if existing_dir is not None and (existing_dir / "skel_male.pkl").exists():
        return existing_dir

    raise FileNotFoundError(f"Expected SKEL model files were not found in {output_dir}")


def download_skel_assets(
    output_dir: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Path]:
    """Download SKEL and return its configured asset paths."""
    directory = download_skel(output_dir=output_dir, username=username, password=password)
    return {
        "skel-female": directory / "skel_female.pkl",
        "skel-male": directory / "skel_male.pkl",
    }


def _download_files(
    name: str,
    url: str,
    files: dict[str, str],
    output_dir: Path,
    username: str | None,
    password: str | None,
) -> dict[str, Path]:
    paths = _find_model_files(name, files, output_dir)
    if paths.keys() != files.keys():
        _fetch(name, url, output_dir, username, password)
        paths = _find_model_files(name, files, output_dir)
    missing = files.keys() - paths.keys()
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Expected {name} model files were not found in {output_dir}: {missing_names}")
    return paths


def _find_model_files(name: str, files: dict[str, str], output_dir: Path) -> dict[str, Path]:
    paths = {}
    for model, pattern in files.items():
        matches = sorted(path for path in output_dir.rglob(pattern) if path.is_file())
        if len(matches) > 1:
            raise ValueError(f"Ambiguous {name} asset {pattern!r} in {output_dir}: {matches}")
        if matches:
            paths[model] = matches[0]
    return paths
