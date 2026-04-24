from . import anny, flame, garment_measurements, mhr, skel, smpl, smplx, soma
from .base import BodyModel

__all__ = [
    # Submodules
    "anny",
    "flame",
    "garment_measurements",
    "mhr",
    "skel",
    "smpl",
    "smplx",
    "soma",
    # Base class
    "BodyModel",
]


def main() -> None:
    import os
    from typing import Annotated, Literal

    import typer

    from .anny.io import download_model as download_anny_model
    from . import fetch
    from .config import CONFIG_FILE, MODELS, get_model_path, set_model_path, unset_model_path
    from .garment_measurements.io import download_model as download_garment_measurements_model
    from .mhr.io import download_model as download_mhr_model
    from .soma.io import download_model as download_soma_model

    Model = Literal[
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
        "soma",
        "garment-measurements",
    ]
    app = typer.Typer(add_completion=False)

    @app.callback(invoke_without_command=True)
    def config(ctx: typer.Context):
        """Show config file path and current settings."""
        if ctx.invoked_subcommand is None:
            print(f"Config file: {CONFIG_FILE}\n")
            print("Current settings:")
            for model in MODELS:
                path = get_model_path(model)
                print(f"  {model}: {path or '(not set)'}")

    @app.command()
    def set(model: Annotated[Model, typer.Argument()], path: str):
        """Set model path."""
        set_model_path(model, path)
        print(f"Set {model} = {path}")

    @app.command()
    def unset(model: Annotated[Model, typer.Argument()]):
        """Remove model path from config."""
        unset_model_path(model)
        print(f"Removed {model} from config")

    @app.command()
    def download(
        model: Annotated[
            Literal["smpl", "smplx", "skel", "flame", "anny", "mhr", "soma", "garment-measurements", "all"],
            typer.Argument(),
        ] = "all",
    ):
        """Download model weights and save their paths."""
        if model in ("smpl", "all"):
            username = os.getenv("SMPL_USERNAME")
            password = os.getenv("SMPL_PASSWORD")
            if username is None or password is None:
                typer.echo("SMPL account: https://smpl.is.tue.mpg.de/")
                username = typer.prompt("Username (SMPL)")
                password = typer.prompt("Password (SMPL)", hide_input=True)
            paths = fetch.download_smpl(username=username, password=password)
            for key, path in sorted(paths.items()):
                set_model_path(key, str(path))
                print(f"Set {key} = {path}")

        if model in ("smplx", "all"):
            username = os.getenv("SMPLX_USERNAME")
            password = os.getenv("SMPLX_PASSWORD")
            if username is None or password is None:
                typer.echo("SMPL-X account: https://smpl-x.is.tue.mpg.de/")
                username = typer.prompt("Username (SMPL-X)")
                password = typer.prompt("Password (SMPL-X)", hide_input=True)
            paths = fetch.download_smplx(username=username, password=password)
            for key, path in sorted(paths.items()):
                set_model_path(key, str(path))
                print(f"Set {key} = {path}")

        if model in ("flame", "all"):
            username = os.getenv("FLAME_USERNAME")
            password = os.getenv("FLAME_PASSWORD")
            if username is None or password is None:
                typer.echo("FLAME account: https://flame.is.tue.mpg.de/")
                username = typer.prompt("Username (FLAME)")
                password = typer.prompt("Password (FLAME)", hide_input=True)
            path = fetch.download_flame(username=username, password=password)
            set_model_path("flame", str(path))
            print(f"Set flame = {path}")

        if model in ("skel", "all"):
            username = os.getenv("SKEL_USERNAME")
            password = os.getenv("SKEL_PASSWORD")
            if username is None or password is None:
                typer.echo("SKEL account: https://skel.is.tue.mpg.de/")
                username = typer.prompt("Username (SKEL)")
                password = typer.prompt("Password (SKEL)", hide_input=True)
            path = fetch.download_skel(username=username, password=password)
            set_model_path("skel", str(path))
            print(f"Set skel = {path}")

        if model in ("anny", "all"):
            path = download_anny_model()
            set_model_path("anny", str(path))
            print(f"Set anny = {path}")

        if model in ("mhr", "all"):
            path = download_mhr_model()
            set_model_path("mhr", str(path))
            print(f"Set mhr = {path}")

        if model in ("soma", "all"):
            path = download_soma_model()
            set_model_path("soma", str(path))
            print(f"Set soma = {path}")

        if model in ("garment-measurements", "all"):
            path = download_garment_measurements_model()
            print(f"Downloaded GarmentMeasurements upstream data to {path}")
            print(
                "Generate garment_measurements.npz from template/male.fbx and then set "
                "garment-measurements to that asset directory."
            )

    app()
