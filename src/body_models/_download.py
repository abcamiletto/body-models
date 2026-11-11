"""Asset downloads: credentialed MPI archives and the public Hugging Face repo."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.error import HTTPError

from huggingface_hub import hf_hub_download

from ._cache import extract_archive, get_cache_dir
from ._catalog import FAMILIES
from ._config import PathLike

HF_MODEL_REPO_ID = "abcamiletto/body-models"


def download_public_assets(package: str, output_dir: PathLike | None, validate: Callable[[Path], Path]) -> Path:
    """
    Return validated public assets, downloading ``<package>/assets.zip`` unless already present.

    Assets count as present when ``validate`` accepts ``output_dir`` (default:
    the ``<package>`` cache directory) without raising ``FileNotFoundError``.
    """
    output_dir = Path(output_dir) if output_dir is not None else get_cache_dir() / package
    try:
        return validate(output_dir)
    except FileNotFoundError:
        pass
    print(f"Downloading {package} assets to {output_dir}...")
    archive = hf_hub_download(HF_MODEL_REPO_ID, f"{package}/assets.zip", cache_dir=get_cache_dir() / "huggingface")
    extract_archive(Path(archive), output_dir)
    print("Done")
    return validate(output_dir)


def download_licensed(
    name: str,
    output_dir: PathLike | None = None,
    username: str | None = None,
    password: str | None = None,
) -> dict[str, Path]:
    """Return a licensed family's asset files by config key, downloading its archive if any are missing."""
    family = FAMILIES[name]
    archive = family.licensed
    if archive is None:
        raise ValueError(f"{name} has no licensed archive.")
    output_dir = Path(output_dir) if output_dir is not None else get_cache_dir() / family.package
    paths = _find_files(name, archive.files, output_dir)
    if paths.keys() != archive.files.keys():
        _fetch(archive.account, archive.url, output_dir, username, password)
        paths = _find_files(name, archive.files, output_dir)
    missing = archive.files.keys() - paths.keys()
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Expected {name} model files were not found in {output_dir}: {missing_names}")
    return paths


def _find_files(name: str, files: Mapping[str, str], output_dir: Path) -> dict[str, Path]:
    paths = {}
    for key, pattern in files.items():
        matches = sorted(path for path in output_dir.rglob(pattern) if path.is_file())
        if len(matches) > 1:
            raise ValueError(f"Ambiguous {name} asset {pattern!r} in {output_dir}: {matches}")
        if matches:
            paths[key] = matches[0]
    return paths


def _fetch(account: str, url: str, output_dir: Path, username: str | None, password: str | None) -> None:
    if username is None or password is None:
        raise ValueError(f"{account} credentials are required to download the model.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        archive_path = Path(temporary) / "archive"
        _download_archive(url, archive_path, username, password)
        extract_archive(archive_path, output_dir)


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
