from pathlib import Path

from .. import config


def get_model_path(model_path: Path | str | None, gender: str) -> Path:
    if model_path is None:
        model_path = config.get_model_path("smpl")

    if model_path is None:
        raise FileNotFoundError(
            "SMPL model not found. Download from https://smpl.is.tue.mpg.de/ "
            "and run: body-models set smpl /path/to/smpl"
        )

    model_path = Path(model_path)

    if model_path.suffix in (".pkl", ".npz") and model_path.exists():
        return model_path

    if model_path.is_dir():
        candidates = [
            model_path / f"SMPL_{gender.upper()}.npz",
            model_path / f"SMPL_{gender.upper()}.pkl",
            model_path / f"basicmodel_{gender[0]}_lbs_10_207_0_v1.1.0.pkl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"SMPL {gender} model not found in {model_path}")

    raise FileNotFoundError(f"SMPL model path {model_path} does not exist")
