from . import anny, flame, mhr, skel, smpl, smplx
from .base import BodyModel

__all__ = [
    # Submodules
    "anny",
    "flame",
    "mhr",
    "skel",
    "smpl",
    "smplx",
    # Base class
    "BodyModel",
]


def main() -> None:
    from typing import Annotated, Literal

    import typer

    from .config import CONFIG_FILE, MODELS, get_model_path, set_model_path, unset_model_path

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

    app()
