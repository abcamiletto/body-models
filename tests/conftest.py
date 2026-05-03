"""pytest configuration for body-models tests."""

import pytest

from body_models import config

import model_assets


@pytest.fixture(autouse=True)
def setup_model_paths(monkeypatch):
    """Use configured model paths, then test assets."""
    get_config_model_path = config.get_model_path

    def get_model_path(model: str):
        model_path = get_config_model_path(model)
        if model_path is not None:
            return model_path

        if model in model_assets.CONFIG_KEYS.values():
            return model_assets.get_test_model_file_for_config_key(model)

        return None

    monkeypatch.setattr(config, "get_model_path", get_model_path)
