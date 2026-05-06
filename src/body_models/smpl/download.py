from pathlib import Path
import shutil
import urllib.parse
import urllib.request
from urllib.error import HTTPError
import zipfile

from body_models.cache import get_cache_dir

SMPL_URL = "https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip"

SMPL_FILES = {
    "smpl-neutral": "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
    "smpl-female": "basicmodel_f_lbs_10_207_0_v1.1.0.pkl",
    "smpl-male": "basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
}

__all__ = ["download_smpl", "download_model", "download_zip"]


def download_smpl(
    cache_dir: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Path]:
    return download_model(
        url=SMPL_URL,
        files=SMPL_FILES,
        cache_dir=Path(cache_dir) if cache_dir else get_cache_dir() / "smpl",
        archive_name="smpl.zip",
        name="SMPL",
        username=username,
        password=password,
    )


def download_model(
    url: str,
    files: dict[str, str],
    cache_dir: Path,
    username: str | None = None,
    password: str | None = None,
    archive_name: str = "model.zip",
    name: str = "model",
) -> dict[str, Path]:
    paths = {model: next(cache_dir.rglob(file_name), None) for model, file_name in files.items()}
    if None not in paths.values():
        return {model: path for model, path in paths.items() if path is not None}

    if username is None or password is None:
        raise ValueError(f"{name} credentials are required to download the model.")

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive_path = cache_dir / archive_name
    download_zip(url, archive_path, username, password)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(cache_dir)
    archive_path.unlink(missing_ok=True)

    paths = {model: next(cache_dir.rglob(file_name), None) for model, file_name in files.items()}
    if None in paths.values():
        raise FileNotFoundError(f"Expected {name} model files were not found in {cache_dir}")

    return {model: path for model, path in paths.items() if path is not None}


def download_zip(url: str, archive_path: Path, username: str, password: str) -> None:
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

    if zipfile.is_zipfile(archive_path):
        return

    snippet = archive_path.read_text(errors="ignore")[:200].strip()
    raise RuntimeError(
        "Download failed. Check your credentials and confirm you accepted the model license."
        + (f" Response started with: {snippet!r}" if snippet else "")
    )
