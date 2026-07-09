from pathlib import Path
import zipfile

from huggingface_hub import hf_hub_download
from platformdirs import user_cache_dir

__all__ = [
    "HF_MODEL_REPO_ID",
    "get_cache_dir",
    "get_cached_path",
    "download_hf_archive",
    "extract_zip",
]

HF_MODEL_REPO_ID = "abcamiletto/body-models"


def get_cache_dir() -> Path:
    """Get the body-models cache directory."""
    return Path(user_cache_dir("body-models"))


def get_cached_path(key: str) -> Path | None:
    """Return the cached path matching key, if present."""
    return next(get_cache_dir().rglob(key), None)


def download_hf_archive(filename: str, dest: Path) -> None:
    """Download and extract an archive from the public body-models Hugging Face repo."""
    archive_path = Path(
        hf_hub_download(
            HF_MODEL_REPO_ID,
            filename,
            cache_dir=get_cache_dir() / "huggingface",
        )
    )
    extract_zip(archive_path, dest)


def extract_zip(archive_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(dest)
