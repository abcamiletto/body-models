"""Persistent model-asset path configuration."""

import json
from importlib import import_module
import tomllib
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

from body_models.catalog import ASSET_SPECS

CONFIG_DIR = Path(user_config_dir("body-models"))
CONFIG_FILE = CONFIG_DIR / "config.toml"

MODELS = tuple(ASSET_SPECS)
Config = dict[str, Any]


def get_config() -> Config:
    """Read the user configuration, returning an empty mapping when absent."""
    if not CONFIG_FILE.exists():
        return {}
    return tomllib.loads(CONFIG_FILE.read_text())


def get_model_path(model: str) -> Path | None:
    """Return the configured path for an asset key, if present."""
    path = get_config().get("paths", {}).get(model)
    return Path(path) if path else None


def set_model_path(model: str, path: str | Path) -> None:
    """Validate and store a model asset path."""
    path = str(validate_model_path(model, path))
    config = get_config()
    config.setdefault("paths", {})[model] = path
    _write_config(config)


def unset_model_path(model: str) -> None:
    """Remove a model asset path if it is configured."""
    config = get_config()
    if "paths" in config and model in config["paths"]:
        del config["paths"][model]
        if not config["paths"]:
            del config["paths"]
        _write_config(config)


def validate_model_path(model: str, path: str | Path) -> Path:
    """Validate an asset path with its model-family loader."""
    try:
        spec = ASSET_SPECS[model]
    except KeyError as exc:
        raise ValueError(f"Unknown model asset: {model!r}") from exc
    validate_path = import_module(spec.validation_module).validate_path
    return validate_path(path)


def _write_config(config: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    if config.get("paths"):
        lines.append("[paths]")
        for model, path in sorted(config["paths"].items()):
            lines.append(f"{model} = {json.dumps(path)}")
    CONFIG_FILE.write_text("\n".join(lines) + "\n" if lines else "")
