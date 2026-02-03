from pathlib import Path

from .. import config
from ..utils import download_and_extract, get_cache_dir

ANNY_URL = "https://github.com/naver/anny/archive/refs/heads/main.zip"


def get_model_path(model_path: Path | str | None = None) -> Path:
    if model_path is None:
        model_path = config.get_model_path("anny")

    if model_path is not None:
        model_path = Path(model_path)
        if model_path.exists():
            return model_path
        raise FileNotFoundError(f"ANNY model path {model_path} does not exist")

    cache_path = get_cache_dir() / "anny"
    if (cache_path / "data" / "mpfb2").exists():
        return cache_path

    return download_model()


def download_model() -> Path:
    cache_dir = get_cache_dir() / "anny"
    print(f"Downloading ANNY model to {cache_dir}...")
    download_and_extract(url=ANNY_URL, dest=cache_dir, extract_subdir="anny-main/")
    print("Done")
    return cache_dir
