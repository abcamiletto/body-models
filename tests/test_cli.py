from pathlib import Path

import pytest
from typer.testing import CliRunner

from body_models import _cli, _download
from body_models.anny import _io as anny_io


@pytest.mark.fast
def test_download_accepts_an_explicit_output_directory(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "anny"
    calls = {}

    def download_model(*, output_dir: Path) -> Path:
        calls["download"] = output_dir
        return output_dir

    def set_model_path(model: str, path: Path) -> None:
        calls["config"] = (model, path)

    monkeypatch.setattr(anny_io, "download_model", download_model)
    monkeypatch.setattr(_cli.config, "set_model_path", set_model_path)

    result = CliRunner().invoke(_cli.app, ["download", "anny", "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert calls == {
        "download": output_dir,
        "config": ("anny", output_dir),
    }


@pytest.mark.fast
def test_licensed_download_uses_environment_credentials(tmp_path, monkeypatch) -> None:
    calls = {}
    paths = {"mano-right": tmp_path / "MANO_RIGHT.pkl", "mano-left": tmp_path / "MANO_LEFT.pkl"}

    def download_licensed(name, output_dir, username, password):
        calls["download"] = (name, output_dir, username, password)
        return paths

    monkeypatch.setenv("MANO_USERNAME", "user")
    monkeypatch.setenv("MANO_PASSWORD", "pass")
    monkeypatch.setattr(_download, "download_licensed", download_licensed)
    monkeypatch.setattr(_cli.config, "set_model_path", lambda model, path: calls.setdefault("config", []).append(model))

    result = CliRunner().invoke(_cli.app, ["download", "mano", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert calls == {"download": ("mano", tmp_path, "user", "pass"), "config": ["mano-left", "mano-right"]}
