from __future__ import annotations

import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from .utils import get_cache_dir

SMPL_URL = "https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip"
SMPLX_URL = "https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip"
SKEL_URL = "https://download.is.tue.mpg.de/download.php?domain=skel&resume=1&sfile=skel_models_v1.1.zip&resume=1"
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
FLAME_FILES = ["flame2023.pkl", "FLAME_NEUTRAL.pkl", "generic_model.pkl", "flame2023_no_jaw.pkl"]


def download_smpl(
    cache_dir: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Path]:
    cache_dir = Path(cache_dir) if cache_dir else get_cache_dir() / "smpl"
    paths = {model: next(cache_dir.rglob(name), None) for model, name in SMPL_FILES.items()}
    if None not in paths.values():
        return {model: path for model, path in paths.items() if path is not None}

    if username is None or password is None:
        raise ValueError("SMPL credentials are required to download the model.")

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive_path = cache_dir / "smpl.zip"
    _download_zip(SMPL_URL, archive_path, username, password)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(cache_dir)
    archive_path.unlink(missing_ok=True)

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

    if username is None or password is None:
        raise ValueError("SMPL-X credentials are required to download the model.")

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive_path = cache_dir / "smplx.zip"
    _download_zip(SMPLX_URL, archive_path, username, password)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(cache_dir)
    archive_path.unlink(missing_ok=True)

    paths = {model: next(cache_dir.rglob(name), None) for model, name in SMPLX_FILES.items()}
    if None in paths.values():
        raise FileNotFoundError(f"Expected SMPL-X model files were not found in {cache_dir}")

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

    if username is None or password is None:
        raise ValueError("FLAME credentials are required to download the model.")

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive_path = cache_dir / "flame.zip"
    _download_zip(FLAME_URL, archive_path, username, password)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(cache_dir)
    archive_path.unlink(missing_ok=True)

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

    if username is None or password is None:
        raise ValueError("SKEL credentials are required to download the model.")

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive_path = cache_dir / "skel.zip"
    _download_zip(SKEL_URL, archive_path, username, password)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(cache_dir)
    archive_path.unlink(missing_ok=True)

    if (cache_dir / "skel_male.pkl").exists():
        return cache_dir
    existing_dir = next(cache_dir.glob("skel_models_v*"), None)
    if existing_dir is not None and (existing_dir / "skel_male.pkl").exists():
        return existing_dir

    raise FileNotFoundError(f"Expected SKEL model files were not found in {cache_dir}")


def _download_zip(url: str, archive_path: Path, username: str, password: str) -> None:
    post_data = urllib.parse.urlencode({"username": username, "password": password}).encode()
    request = urllib.request.Request(url, data=post_data)

    with urllib.request.urlopen(request) as response, archive_path.open("wb") as f:
        shutil.copyfileobj(response, f)

    if zipfile.is_zipfile(archive_path):
        return

    snippet = archive_path.read_text(errors="ignore")[:200].strip()
    raise RuntimeError(
        "Download failed. Check your credentials and confirm you accepted the model license."
        + (f" Response started with: {snippet!r}" if snippet else "")
    )
