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
    username: str,
    password: str,
) -> dict[str, Path]:
    return download_model(
        url=SMPL_URL,
        files=SMPL_FILES,
        output_file=get_cache_dir() / "smpl" / "smpl.zip",
        name="SMPL",
        username=username,
        password=password,
    )


def download_model(
    url: str,
    files: dict[str, str],
    output_file: Path,
    username: str,
    password: str,
    name: str = "model",
) -> dict[str, Path]:
    output_file = Path(output_file)
    output_dir = output_file.parent
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    download_zip(url, output_file, username, password)
    with zipfile.ZipFile(output_file) as zf:
        zf.extractall(output_dir)
    output_file.unlink(missing_ok=True)

    output_paths = {}
    for model, file_name in files.items():
        output_path = next(output_dir.rglob(file_name), None)
        if output_path is None:
            raise FileNotFoundError(f"Expected {name} file {file_name!r} was not found in {output_dir}")
        output_paths[model] = output_path

    return output_paths


def download_zip(url: str, output_file: Path, username: str, password: str) -> None:
    post_data = urllib.parse.urlencode({"username": username, "password": password}).encode()
    request = urllib.request.Request(url, data=post_data)

    try:
        with urllib.request.urlopen(request) as response, output_file.open("wb") as f:
            shutil.copyfileobj(response, f)
    except HTTPError as exc:
        snippet = exc.read(200).decode(errors="ignore").strip()
        raise RuntimeError(
            f"Download failed with HTTP {exc.code}. Check your credentials and confirm you accepted the model license."
            + (f" Response started with: {snippet!r}" if snippet else "")
        ) from exc

    if zipfile.is_zipfile(output_file):
        return

    snippet = output_file.read_text(errors="ignore")[:200].strip()
    raise RuntimeError(
        "Download failed. Check your credentials and confirm you accepted the model license."
        + (f" Response started with: {snippet!r}" if snippet else "")
    )
