import tomllib
from pathlib import Path

from platformdirs import user_config_dir

CONFIG_DIR = Path(user_config_dir("body-models"))
CONFIG_FILE = CONFIG_DIR / "config.toml"

MODELS = ["smpl", "smplx", "skel", "anny", "mhr", "flame"]


def get_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    return tomllib.loads(CONFIG_FILE.read_text())


def get_model_path(model: str) -> Path | None:
    path = get_config().get("paths", {}).get(model)
    return Path(path) if path else None


def set_model_path(model: str, path: str) -> None:
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


def _write_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    if config.get("paths"):
        lines.append("[paths]")
        for model, path in sorted(config["paths"].items()):
            lines.append(f'{model} = "{path}"')
    CONFIG_FILE.write_text("\n".join(lines) + "\n" if lines else "")
