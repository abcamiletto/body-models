"""pytest configuration for body-models tests."""

from pathlib import Path

import pytest

from body_models import config

ASSET_DIR = Path(__file__).parent / "assets"


@pytest.fixture(autouse=True)
def setup_model_paths(monkeypatch):
    """Set up model paths to use test assets."""

    # Create a mock config that returns test asset paths
    def mock_get_model_path(model: str) -> Path | None:
        asset_path = ASSET_DIR / model / "model"
        if asset_path.exists():
            return asset_path
        return None

    monkeypatch.setattr(config, "get_model_path", mock_get_model_path)
