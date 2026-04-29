import json
import tomllib
from pathlib import Path

from platformdirs import user_config_dir

CONFIG_DIR = Path(user_config_dir("body-models"))
CONFIG_FILE = CONFIG_DIR / "config.toml"

MODELS = [
    "smpl-male",
    "smpl-female",
    "smpl-neutral",
    "smplx-male",
    "smplx-female",
    "smplx-neutral",
    "smplh-male",
    "smplh-female",
    "smplh-neutral",
    "mano-right",
    "mano-left",
    "skel",
    "anny",
    "mhr",
    "flame",
    "brainco",
    "g1",
    "soma",
    "garment-measurements",
    "myofullbody",
]

SMPL_MODELS = {"smpl-male", "smpl-female", "smpl-neutral"}
SMPLX_MODELS = {"smplx-male", "smplx-female", "smplx-neutral"}
SMPLH_MODELS = {"smplh-male", "smplh-female", "smplh-neutral"}
MANO_MODELS = {"mano-right", "mano-left"}


def get_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    return tomllib.loads(CONFIG_FILE.read_text())


def get_model_path(model: str) -> Path | None:
    path = get_config().get("paths", {}).get(model)
    return Path(path) if path else None


def set_model_path(model: str, path: str | Path) -> None:
    path = str(validate_model_path(model, path))
    config = get_config()
    config.setdefault("paths", {})[model] = path
    _write_config(config)


def unset_model_path(model: str) -> None:
    config = get_config()
    if "paths" in config and model in config["paths"]:
        del config["paths"][model]
        if not config["paths"]:
            del config["paths"]
        _write_config(config)


def validate_model_path(model: str, path: str | Path) -> Path:
    if model in SMPL_MODELS:
        from .smpl.io import validate_path
    elif model in SMPLX_MODELS:
        from .smplx.io import validate_path
    elif model in SMPLH_MODELS:
        from .smplh.io import validate_path
    elif model in MANO_MODELS:
        from .mano.io import validate_path
    elif model == "skel":
        from .skel.io import validate_path
    elif model == "anny":
        from .anny.io import validate_path
    elif model == "mhr":
        from .mhr.io import validate_path
    elif model == "flame":
        from .flame.io import validate_path
    elif model == "brainco":
        from .brainco.io import validate_path
    elif model == "g1":
        from .g1.io import validate_path
    elif model == "soma":
        from .soma.io import validate_path
    elif model == "garment-measurements":
        from .garment_measurements.io import validate_path
    else:
        raise ValueError(f"Unknown model: {model}")

    return validate_path(path)


def _write_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    if config.get("paths"):
        lines.append("[paths]")
        for model, path in sorted(config["paths"].items()):
            lines.append(f"{model} = {json.dumps(path)}")
    CONFIG_FILE.write_text("\n".join(lines) + "\n" if lines else "")
