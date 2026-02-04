from pathlib import Path

from .. import config


def get_model_path(model_path: Path | str | None) -> Path:
    if model_path is None:
        model_path = config.get_model_path("flame")

    if model_path is None:
        raise FileNotFoundError(
            "FLAME model not found. Download from https://flame.is.tue.mpg.de/ "
            "and run: body-models set flame /path/to/flame"
        )

    model_path = Path(model_path)

    if model_path.is_file():
        return model_path

    if model_path.is_dir():
        # Try common FLAME model filenames
        for candidate_name in ["FLAME_NEUTRAL.pkl", "FLAME_NEUTRAL.npz", "flame2023.pkl", "generic_model.pkl"]:
            candidate = model_path / candidate_name
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"FLAME model not found in {model_path}")
