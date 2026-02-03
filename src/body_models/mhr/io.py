from pathlib import Path

from .. import config
from ..utils import download_and_extract, get_cache_dir

MHR_URL = "https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip"


def get_model_path(model_path: Path | str | None = None) -> Path:
    if model_path is None:
        model_path = config.get_model_path("mhr")

    if model_path is not None:
        model_path = Path(model_path)
        if (model_path / "mhr_model.pt").exists():
            return model_path
        if model_path.exists():
            return model_path.parent
        raise FileNotFoundError(f"MHR model path {model_path} does not exist")

    cache_path = get_cache_dir() / "mhr"
    if (cache_path / "mhr_model.pt").exists():
        return cache_path

    return download_model()


def download_model() -> Path:
    cache_dir = get_cache_dir() / "mhr"
    print(f"Downloading MHR model to {cache_dir}...")
    download_and_extract(url=MHR_URL, dest=cache_dir, extract_subdir="assets/")
    print("Done")
    return cache_dir
