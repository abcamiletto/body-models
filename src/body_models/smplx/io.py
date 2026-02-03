from pathlib import Path

from .. import config


def get_model_path(model_path: Path | str | None, gender: str) -> Path:
    if model_path is None:
        model_path = config.get_model_path("smplx")

    if model_path is None:
        raise FileNotFoundError(
            "SMPLX model not found. Download from https://smpl-x.is.tue.mpg.de/ "
            "and run: body-models set smplx /path/to/smplx"
        )

    model_path = Path(model_path)

    if model_path.is_file():
        return model_path

    if model_path.is_dir():
        candidate = model_path / f"SMPLX_{gender.upper()}.npz"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"SMPLX {gender} model not found in {model_path}")
