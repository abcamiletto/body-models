"""Command-line configuration and asset management."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import typer

from body_models import _config as config
from body_models import _download
from body_models._cache import get_cache_dir
from body_models._catalog import FAMILIES

app = typer.Typer(add_completion=False, no_args_is_help=False)

DOWNLOAD_NAMES = tuple(FAMILIES)


@app.callback(invoke_without_command=True)
def show_config(ctx: typer.Context) -> None:
    """Configure and download body-model assets."""
    if ctx.invoked_subcommand is not None:
        return
    typer.echo(f"Config file: {config.CONFIG_FILE}\n")
    typer.echo(f"Asset cache: {get_cache_dir()}\n")
    typer.echo("Current settings:")
    for model in config.ASSET_KEYS:
        typer.echo(f"  {model}: {config.get_model_path(model) or '(not set)'}")


@app.command()
def set(model: Annotated[str, typer.Argument()], path: Path) -> None:
    """Validate and save a model asset path."""
    _require_choice(model, config.ASSET_KEYS, "model asset")
    config.set_model_path(model, path)
    typer.echo(f"Set {model} = {config.get_model_path(model)}")


@app.command()
def unset(model: Annotated[str, typer.Argument()]) -> None:
    """Remove a model asset path from the config."""
    _require_choice(model, config.ASSET_KEYS, "model asset")
    config.unset_model_path(model)
    typer.echo(f"Removed {model} from config")


@app.command()
def download(
    model: Annotated[str, typer.Argument()] = "all",
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Exact destination for one model, or parent directory for all models."),
    ] = None,
) -> None:
    """Download one model family, or every supported family."""
    _require_choice(model, (*DOWNLOAD_NAMES, "all"), "download")
    names = DOWNLOAD_NAMES if model == "all" else (model,)
    for name in names:
        destination = output_dir / name if output_dir is not None and model == "all" else output_dir
        _download_family(name, destination)


@app.command("preprocess-soma")
def preprocess_soma(upstream_dir: Path, output_dir: Path) -> None:
    """Convert upstream SOMA-X 0.2.1 files into body-models assets."""
    from body_models.soma._io import preprocess_model

    path = preprocess_model(upstream_dir, output_dir)
    _save_paths({"soma": path})


def _download_family(name: str, output_dir: Path | None) -> None:
    family = FAMILIES[name]
    if family.licensed is None:
        _save_paths({name: family.io().download_model(output_dir=output_dir)})
        return
    username, password = _credentials(family.licensed.account, family.licensed.site)
    _save_paths(_download.download_licensed(name, output_dir, username, password))


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
