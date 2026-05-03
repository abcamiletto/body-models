from pathlib import Path
from typing import Literal


from .. import config
from ..common import simplify_mesh

PathLike = Path | str

__all__ = ["get_model_path", "simplify_mesh"]


def validate_path(model_path: PathLike) -> Path:
    model_path = Path(model_path)
    if model_path.is_dir():
        raise ValueError(f"Expected a SKEL model file, got directory: {model_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"SKEL model file not found: {model_path}")
    return model_path


def get_model_path(model_path: PathLike | None, gender: Literal["male", "female"] | None) -> Path:
    if gender is None:
        raise ValueError("gender must be 'male' or 'female'.")

    if model_path is None:
        model_path = config.get_model_path(f"skel-{gender.lower()}")

    if model_path is None:
        raise FileNotFoundError(
            "SKEL model not found. Download from https://skel.is.tue.mpg.de/ "
            f"and run: body-models set skel-{gender.lower()} /path/to/model.pkl"
        )

    return validate_path(model_path)
