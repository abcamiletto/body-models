from pathlib import Path

from platformdirs import user_cache_dir

__all__ = ["get_cache_dir", "get_cached_path", "download_file", "download_and_extract"]

HF_DATASET_BASE_URL = "https://huggingface.co/datasets/abcamiletto/body-models-assets/resolve/main/models_hub"


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


def download_and_extract(
    url: str,
    dest: Path,
    extract_subdir: str | None = None,
) -> None:
    """Download a zip file and extract it to dest.

    Args:
        url: URL to download from.
        dest: Destination directory for extracted files.
        extract_subdir: If specified, only extract files from this subdirectory
            within the zip archive. The subdirectory prefix is stripped.
    """
    import tempfile
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        download_file(url, tmp_path)

        with zipfile.ZipFile(tmp_path) as zf:
            if extract_subdir is None:
                zf.extractall(dest)
            else:
                # Normalize the subdir path
                if not extract_subdir.endswith("/"):
                    extract_subdir = extract_subdir + "/"

                for member in zf.namelist():
                    if member.startswith(extract_subdir):
                        # Strip the subdirectory prefix
                        relative_path = member[len(extract_subdir) :]
                        if not relative_path:
                            continue

                        target_path = dest / relative_path

                        if member.endswith("/"):
                            target_path.mkdir(parents=True, exist_ok=True)
                        else:
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(member) as src, open(target_path, "wb") as dst:
                                dst.write(src.read())
    finally:
        tmp_path.unlink(missing_ok=True)
