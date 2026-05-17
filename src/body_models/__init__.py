from . import anny, brainco, flame, g1, garment_measurements, mano, mhr, myofullbody, skel, smpl, smplh, smplx, soma
from .base import BodyModel, KinematicJoint
from .constants import Joint

__all__ = [
    # Submodules
    "anny",
    "brainco",
    "flame",
    "garment_measurements",
    "g1",
    "mano",
    "mhr",
    "myofullbody",
    "skel",
    "smpl",
    "smplh",
    "smplx",
    "soma",
    # Base class
    "BodyModel",
    "KinematicJoint",
    "Joint",
]


def main() -> None:
    import os
    from typing import Annotated, Literal

    import typer

    from . import download as official_downloads
    from .anny.io import download_model as download_anny_model
    from .brainco.io import download_model as download_brainco_model
    from .config import CONFIG_FILE, MODELS, get_model_path, set_model_path, unset_model_path
    from .g1.io import download_model as download_g1_model
    from .garment_measurements.io import download_model as download_garment_measurements_model
    from .mhr.io import download_model as download_mhr_model
    from .myofullbody.io import download_model as download_myofullbody_model
    from .smpl.download import download_smpl
    from .soma.io import download_model as download_soma_model

    Model = Literal[
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
        "skel-male",
        "skel-female",
        "anny",
        "mhr",
        "flame",
        "brainco",
        "g1",
        "soma",
        "garment-measurements",
        "myofullbody",
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
            Literal[
                "smpl",
                "smplh",
                "mano",
                "smplx",
                "skel",
                "flame",
                "anny",
                "brainco",
                "mhr",
                "g1",
                "soma",
                "garment-measurements",
                "myofullbody",
                "all",
            ],
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
            paths = download_smpl(username=username, password=password)
            for key, path in sorted(paths.items()):
                set_model_path(key, str(path))
                print(f"Set {key} = {path}")

        if model in ("smplh", "all"):
            username = os.getenv("SMPLH_USERNAME")
            password = os.getenv("SMPLH_PASSWORD")
            if username is None or password is None:
                typer.echo("SMPL-H account: https://mano.is.tue.mpg.de/")
                username = typer.prompt("Username (SMPL-H)")
                password = typer.prompt("Password (SMPL-H)", hide_input=True)
            paths = official_downloads.download_smplh(username=username, password=password)
            for key, path in sorted(paths.items()):
                set_model_path(key, str(path))
                print(f"Set {key} = {path}")

        if model in ("mano", "all"):
            username = os.getenv("MANO_USERNAME")
            password = os.getenv("MANO_PASSWORD")
            if username is None or password is None:
                typer.echo("MANO account: https://mano.is.tue.mpg.de/")
                username = typer.prompt("Username (MANO)")
                password = typer.prompt("Password (MANO)", hide_input=True)
            paths = official_downloads.download_mano(username=username, password=password)
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
            paths = official_downloads.download_smplx(username=username, password=password)
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
            path = official_downloads.download_flame(username=username, password=password)
            set_model_path("flame", str(path))
            print(f"Set flame = {path}")

        if model in ("skel", "all"):
            username = os.getenv("SKEL_USERNAME")
            password = os.getenv("SKEL_PASSWORD")
            if username is None or password is None:
                typer.echo("SKEL account: https://skel.is.tue.mpg.de/")
                username = typer.prompt("Username (SKEL)")
                password = typer.prompt("Password (SKEL)", hide_input=True)
            path = official_downloads.download_skel(username=username, password=password)
            skel_paths = {
                "skel-female": path / "skel_female.pkl",
                "skel-male": path / "skel_male.pkl",
            }
            for key, model_path in sorted(skel_paths.items()):
                set_model_path(key, str(model_path))
                print(f"Set {key} = {model_path}")

        if model in ("anny", "all"):
            path = download_anny_model()
            set_model_path("anny", str(path))
            print(f"Set anny = {path}")

        if model in ("brainco", "all"):
            path = download_brainco_model()
            set_model_path("brainco", str(path))
            print(f"Set brainco = {path}")

        if model in ("mhr", "all"):
            path = download_mhr_model()
            set_model_path("mhr", str(path))
            print(f"Set mhr = {path}")

        if model in ("g1", "all"):
            path = download_g1_model()
            set_model_path("g1", str(path))
            print(f"Set g1 = {path}")

        if model in ("soma", "all"):
            path = download_soma_model()
            set_model_path("soma", str(path))
            print(f"Set soma = {path}")

        if model in ("garment-measurements", "all"):
            path = download_garment_measurements_model()
            set_model_path("garment-measurements", str(path))
            print(f"Set garment-measurements = {path}")

        if model in ("myofullbody", "all"):
            path = download_myofullbody_model()
            set_model_path("myofullbody", str(path))
            print(f"Set myofullbody = {path}")

    app()
