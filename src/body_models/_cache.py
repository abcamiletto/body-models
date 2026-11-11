import hashlib
import json
import tarfile
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from platformdirs import user_cache_dir

__all__ = [
    "derived_cache_key",
    "extract_archive",
    "get_cache_dir",
    "write_npz_atomic",
]


def derived_cache_key(
    version: str,
    *,
    sources: Iterable[Path],
    parameters: Iterable[str | int | float | bool | None] = (),
) -> str:
    """Identify a derived artifact by its transformation and source files."""
    digest = hashlib.sha256()
    digest.update(version.encode() + b"\0")
    digest.update(json.dumps(tuple(parameters), separators=(",", ":")).encode() + b"\0")
    for source in sources:
        source = source.resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"Cache key source must be a file: {source}")
        digest.update(str(source).encode() + b"\0")
        stat = source.stat()
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode() + b"\0")
    return digest.hexdigest()[:32]


def get_cache_dir() -> Path:
    """Get the body-models cache directory."""
    return Path(user_cache_dir("body-models"))


def write_npz_atomic(
    path: Path,
    /,
    *,
    compressed: bool = True,
    **arrays: Any,
) -> None:
    """Write an NPZ cache completely before replacing its destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if compressed else np.savez
    prefix = f".{path.name}-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=path.parent) as temporary:
        temporary_path = Path(temporary) / "data.npz"
        save(temporary_path, **arrays)
        temporary_path.replace(path)


def extract_archive(archive_path: Path, dest: Path) -> None:
    """Extract an archive completely before replacing its destination."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{dest.name}-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=dest.parent) as temporary:
        temporary_dir = Path(temporary)
        contents = temporary_dir / "contents"
        contents.mkdir()

        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                _validate_paths(archive.namelist())
                archive.extractall(contents)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as archive:
                members = archive.getmembers()
                _validate_paths(member.name for member in members)
                archive.extractall(contents, members=members, filter="data")
        else:
            raise ValueError(f"Unsupported archive: {archive_path}")

        previous = temporary_dir / "previous"
        if dest.exists():
            dest.rename(previous)
        try:
            contents.rename(dest)
        except OSError:
            if previous.exists():
                previous.rename(dest)
            raise


def _validate_paths(names: Iterable[str]) -> None:
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe path in archive: {name}")
