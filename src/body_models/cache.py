from pathlib import Path

from platformdirs import user_cache_dir

__all__ = [
    "HF_MODEL_BASE_URL",
    "get_cache_dir",
    "get_cached_path",
    "download_file",
    "download_and_extract",
]

HF_MODEL_BASE_URL = "https://huggingface.co/abcamiletto/body-models/resolve/main"


def get_cache_dir() -> Path:
    """Get the body-models cache directory."""
    return Path(user_cache_dir("body-models"))


def get_cached_path(key: str) -> Path | None:
    """Return the cached path matching key, if present."""
    return next(get_cache_dir().rglob(key), None)


def download_file(url: str, dest: Path) -> None:
    """Download a single file to ``dest``.

    ``HF_TOKEN`` is honored for private Hugging Face dataset assets.
    """
    import os
    import shutil
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url)
    if token := os.getenv("HF_TOKEN"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request) as src, dest.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def download_and_extract(url: str, dest: Path) -> None:
    """Download a zip file and extract it to dest."""
    import tempfile
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        download_file(url, tmp_path)

        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(dest)
    finally:
        tmp_path.unlink(missing_ok=True)
