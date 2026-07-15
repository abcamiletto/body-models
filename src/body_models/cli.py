"""Command-line configuration and asset management."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from body_models import config

app = typer.Typer(add_completion=False, no_args_is_help=False)

DOWNLOAD_NAMES = (
    "smpl",
    "smplh",
    "mano",
    "smplx",
    "smpl-humanoid",
    "skel",
    "flame",
    "anny",
    "brainco",
    "mhr",
    "g1",
    "soma",
    "garment-measurements",
    "myofullbody",
)


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
    if name in _REGISTERED_DOWNLOADS:
        account, url, downloader = _REGISTERED_DOWNLOADS[name]
        username, password = _credentials(account, url)
        _save_paths(downloader(username=username, password=password))
        return

    if name == "skel":
        from body_models import download as downloads

        username, password = _credentials("SKEL", "https://skel.is.tue.mpg.de/")
        directory = downloads.download_skel(username=username, password=password)
        _save_paths(
            {
                "skel-female": directory / "skel_female.pkl",
                "skel-male": directory / "skel_male.pkl",
            }
        )
        return

    if name == "smpl-humanoid":
        from body_models.robots.smpl_humanoid.constants import SMPL_HUMANOID_VARIANTS
        from body_models.robots.smpl_humanoid.io import download_model

        _save_paths({f"smpl-humanoid-{source}": download_model(source) for source in SMPL_HUMANOID_VARIANTS})
        return

    downloader, config_key = _public_download(name)
    _save_paths({config_key: downloader()})


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


def _public_download(name: str) -> tuple[Callable[[], Path], str]:
    if name == "anny":
        from body_models.bodies.anny.io import download_model
    elif name == "brainco":
        from body_models.robots.brainco.io import download_model
    elif name == "mhr":
        from body_models.bodies.mhr.io import download_model
    elif name == "g1":
        from body_models.robots.g1.io import download_model
    elif name == "soma":
        from body_models.bodies.soma.io import download_model
    elif name == "garment-measurements":
        from body_models.bodies.garment_measurements.io import download_model
    elif name == "myofullbody":
        from body_models.skeletons.myofullbody.io import download_model
    else:
        raise ValueError(f"No public downloader for {name!r}")
    return download_model, name


def _require_choice(value: str, choices: tuple[str, ...], label: str) -> None:
    if value not in choices:
        expected = ", ".join(choices)
        raise typer.BadParameter(f"Unknown {label} {value!r}. Expected one of: {expected}")


def _official_downloads() -> dict[str, tuple[str, str, Callable[..., Mapping[str, Path]]]]:
    from body_models import download as downloads

    return {
        "smpl": ("SMPL", "https://smpl.is.tue.mpg.de/", downloads.download_smpl),
        "smplh": ("SMPLH", "https://mano.is.tue.mpg.de/", downloads.download_smplh),
        "mano": ("MANO", "https://mano.is.tue.mpg.de/", downloads.download_mano),
        "smplx": ("SMPLX", "https://smpl-x.is.tue.mpg.de/", downloads.download_smplx),
        "flame": ("FLAME", "https://flame.is.tue.mpg.de/", _download_flame),
    }


def _download_flame(**credentials: Any) -> Mapping[str, Path]:
    from body_models import download as downloads

    return {"flame": downloads.download_flame(**credentials)}


_REGISTERED_DOWNLOADS = _official_downloads()


def main() -> None:
    app()


__all__ = ["app", "main"]
