"""Command-line configuration and asset management."""

from __future__ import annotations

import os
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Annotated

import typer

from body_models import config
from body_models.catalog import DOWNLOAD_SPECS

app = typer.Typer(add_completion=False, no_args_is_help=False)

DOWNLOAD_NAMES = tuple(DOWNLOAD_SPECS)


@app.callback(invoke_without_command=True)
def show_config(ctx: typer.Context) -> None:
    """Configure and download body-model assets."""
    if ctx.invoked_subcommand is not None:
        return
    typer.echo(f"Config file: {config.CONFIG_FILE}\n")
    typer.echo("Current settings:")
    for model in config.MODELS:
        typer.echo(f"  {model}: {config.get_model_path(model) or '(not set)'}")


@app.command()
def set(model: Annotated[str, typer.Argument()], path: Path) -> None:
    """Validate and save a model asset path."""
    _require_choice(model, config.MODELS, "model asset")
    config.set_model_path(model, path)
    typer.echo(f"Set {model} = {config.get_model_path(model)}")


@app.command()
def unset(model: Annotated[str, typer.Argument()]) -> None:
    """Remove a model asset path from the config."""
    _require_choice(model, config.MODELS, "model asset")
    config.unset_model_path(model)
    typer.echo(f"Removed {model} from config")


@app.command()
def download(model: Annotated[str, typer.Argument()] = "all") -> None:
    """Download one model family, or every supported family."""
    _require_choice(model, (*DOWNLOAD_NAMES, "all"), "download")
    names = DOWNLOAD_NAMES if model == "all" else (model,)
    for name in names:
        _download(name)


@app.command("preprocess-soma")
def preprocess_soma(upstream_dir: Path, output_dir: Path) -> None:
    """Normalize upstream SOMA-X 0.2.1 assets for runtime use."""
    from body_models.bodies.soma.io import preprocess_model

    path = preprocess_model(upstream_dir, output_dir)
    _save_paths({"soma": path})


def _download(name: str) -> None:
    spec = DOWNLOAD_SPECS[name]
    downloader = getattr(import_module(spec.module), spec.function)
    kwargs = {}
    if spec.credentials is not None:
        username, password = _credentials(spec.credentials.account, spec.credentials.url)
        kwargs = {"username": username, "password": password}
    result = downloader(**kwargs)
    if spec.output_key is not None:
        _save_paths({spec.output_key: result})
    elif isinstance(result, Mapping):
        _save_paths(result)
    else:
        raise TypeError(f"{spec.module}.{spec.function} must return a mapping")


def _credentials(account: str, url: str) -> tuple[str, str]:
    username = os.getenv(f"{account}_USERNAME")
    password = os.getenv(f"{account}_PASSWORD")
    if username is None or password is None:
        typer.echo(f"{account} account: {url}")
        username = typer.prompt(f"Username ({account})")
        password = typer.prompt(f"Password ({account})", hide_input=True)
    return username, password


def _save_paths(paths: Mapping[str, str | Path]) -> None:
    for key, path in sorted(paths.items()):
        config.set_model_path(key, path)
        typer.echo(f"Set {key} = {path}")


def _require_choice(value: str, choices: tuple[str, ...], label: str) -> None:
    if value not in choices:
        expected = ", ".join(choices)
        raise typer.BadParameter(f"Unknown {label} {value!r}. Expected one of: {expected}")


def main() -> None:
    app()


__all__ = ["app", "main"]
