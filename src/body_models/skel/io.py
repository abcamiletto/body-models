from pathlib import Path
from typing import Literal


from .. import config
from ..common import simplify_mesh

PathLike = Path | str

__all__ = ["get_model_path", "simplify_mesh"]
SKELGender = Literal["male", "female"]


def validate_path(model_path: PathLike) -> Path:
    model_path = Path(model_path)
    if model_path.is_file():
        raise ValueError(f"Expected a SKEL model directory, got file: {model_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"SKEL model directory not found: {model_path}")
    missing = [name for name in ("skel_male.pkl", "skel_female.pkl") if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"SKEL model directory {model_path} is missing required files: {', '.join(missing)}")
    return model_path


def get_model_path(model_path: PathLike | None, gender: SKELGender | None) -> Path:
    if gender is None:
        raise ValueError("gender must be 'male' or 'female'.")

    if model_path is None:
        model_path = config.get_model_path("skel")

    if model_path is None:
        raise FileNotFoundError(
            "SKEL model not found. Download from https://skel.is.tue.mpg.de/ "
            "and run: body-models set skel /path/to/skel (only male/female supported)"
        )

    model_path = validate_path(model_path)
    candidate = model_path / f"skel_{gender.lower()}.pkl"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"SKEL {gender} model not found in {model_path}")
