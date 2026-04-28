import tomllib
from importlib import import_module
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
    "skel",
    "anny",
    "mhr",
    "flame",
    "g1",
    "soma",
    "garment-measurements",
]

_VALIDATORS = {
    "smpl-male": "body_models.smpl.io",
    "smpl-female": "body_models.smpl.io",
    "smpl-neutral": "body_models.smpl.io",
    "smplx-male": "body_models.smplx.io",
    "smplx-female": "body_models.smplx.io",
    "smplx-neutral": "body_models.smplx.io",
    "skel": "body_models.skel.io",
    "anny": "body_models.anny.io",
    "mhr": "body_models.mhr.io",
    "flame": "body_models.flame.io",
    "g1": "body_models.g1.io",
    "soma": "body_models.soma.io",
    "garment-measurements": "body_models.garment_measurements.io",
}


def get_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    return tomllib.loads(CONFIG_FILE.read_text())


def get_model_path(model: str) -> Path | None:
    path = get_config().get("paths", {}).get(model)
    return Path(path) if path else None


def set_model_path(model: str, path: str) -> None:
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
    if model not in _VALIDATORS:
        raise ValueError(f"Unknown model: {model}")
    validator = getattr(import_module(_VALIDATORS[model]), "validate_path")
    return validator(path)


def _write_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    if config.get("paths"):
        lines.append("[paths]")
        for model, path in sorted(config["paths"].items()):
            lines.append(f'{model} = "{path}"')
    CONFIG_FILE.write_text("\n".join(lines) + "\n" if lines else "")
