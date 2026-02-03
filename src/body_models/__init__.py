from . import anny, mhr, skel, smpl, smplx
from .anny import ANNY
from .base import BodyModel
from .mhr import MHR
from .skel import SKEL
from .smpl import SMPL
from .smplx import SMPLX

__all__ = [
    # Submodules (for bm.smplx.pack_pose, etc.)
    "anny",
    "mhr",
    "skel",
    "smpl",
    "smplx",
    # Model classes
    "ANNY",
    "BodyModel",
    "MHR",
    "SKEL",
    "SMPL",
    "SMPLX",
]


def main() -> None:
    from typing import Annotated, Literal

    import typer

    from .config import CONFIG_FILE, MODELS, get_model_path, set_model_path, unset_model_path

    Model = Literal["smpl", "smplx", "skel", "anny", "mhr"]
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
