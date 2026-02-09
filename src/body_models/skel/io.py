from pathlib import Path

import numpy as np

from .. import config
from ..common import simplify_mesh

__all__ = ["get_model_path", "simplify_mesh"]


def get_model_path(model_path: Path | str | None, gender: str) -> Path:
    if model_path is None:
        model_path = config.get_model_path("skel")

    if model_path is None:
        raise FileNotFoundError(
            "SKEL model not found. Download from https://skel.is.tue.mpg.de/ "
            "and run: body-models set skel /path/to/skel (only male/female supported)"
        )

    model_path = Path(model_path)

    if model_path.is_file():
        return model_path

    if model_path.is_dir():
        candidate = model_path / f"skel_{gender.lower()}.pkl"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"SKEL {gender} model not found in {model_path}")


