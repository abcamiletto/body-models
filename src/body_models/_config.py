"""Persistent model-asset path configuration and resolution."""

import json
import tomllib
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

from body_models._catalog import ASSET_FAMILIES, FAMILIES

CONFIG_DIR = Path(user_config_dir("body-models"))
CONFIG_FILE = CONFIG_DIR / "config.toml"

ASSET_KEYS = tuple(ASSET_FAMILIES)
Config = dict[str, Any]
PathLike = Path | str


def get_config() -> Config:
    """Read the user configuration, returning an empty mapping when absent."""
    if not CONFIG_FILE.exists():
        return {}
    return tomllib.loads(CONFIG_FILE.read_text())


def get_model_path(model: str) -> Path | None:
    """Return the configured path for an asset key, if present."""
    path = get_config().get("paths", {}).get(model)
    return Path(path) if path else None


def set_model_path(model: str, path: PathLike) -> None:
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


def validate_model_path(model: str, path: PathLike) -> Path:
    """Validate an asset path with its model-family loader."""
    try:
        family = ASSET_FAMILIES[model]
    except KeyError as exc:
        raise ValueError(f"Unknown model asset: {model!r}") from exc
    return family.io().validate_path(path)


def resolve_model_path(name: str, model_path: PathLike | None = None, variant: str | None = None) -> Path:
    """
    Return a family's validated asset path.

    An explicit ``model_path`` wins. Otherwise ``variant`` selects the config key
    ``<name>-<variant>`` (or ``<name>`` without one). Unconfigured public assets
    are downloaded to the cache; unconfigured licensed assets raise with the CLI
    commands that provide them.
    """
    family = FAMILIES[name]
    io = family.io()
    if model_path is None:
        key = name if variant is None else f"{name}-{variant}"
        model_path = get_model_path(key)
        if model_path is None:
            if family.licensed is None:
                return io.download_model()
            example = family.licensed.files[key]
            raise FileNotFoundError(
                f"No {key} asset is configured. Run `body-models download {name}` or configure an existing file "
                f"with `body-models set {key} /path/to/{example}`."
            )
    return io.validate_path(model_path)


def validate_file(model_path: PathLike, label: str, suffixes: tuple[str, ...]) -> Path:
    """Validate a single-file model asset."""
    model_path = Path(model_path)
    if model_path.is_dir():
        raise ValueError(f"Expected a {label} model file, got directory: {model_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"{label} model file not found: {model_path}")
    if model_path.suffix not in suffixes:
        raise ValueError(f"Expected a {label} {' or '.join(suffixes)} file, got: {model_path}")
    return model_path


def _write_config(config: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    if config.get("paths"):
        lines.append("[paths]")
        for model, path in sorted(config["paths"].items()):
            lines.append(f"{model} = {json.dumps(path)}")
    CONFIG_FILE.write_text("\n".join(lines) + "\n" if lines else "")
