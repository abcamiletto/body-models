"""Public model and asset catalog contracts."""

import pytest

import body_models
from body_models.catalog import ASSET_SPECS, MODEL_SPECS, PUBLIC_MODULES
from body_models.registry import get_model_spec


@pytest.mark.fast
def test_public_modules_are_derived_from_model_catalog() -> None:
    catalog_modules = {spec.public_module for spec in MODEL_SPECS.values()}
    assert set(PUBLIC_MODULES.values()) == catalog_modules
    assert all(
        getattr(body_models, name).__name__.removeprefix("body_models.") == module
        for name, module in PUBLIC_MODULES.items()
    )


@pytest.mark.fast
def test_registry_normalizes_public_names() -> None:
    assert get_model_spec("smpl_humanoid") is MODEL_SPECS["smpl-humanoid"]
    assert get_model_spec(" GARMENT_MEASUREMENTS ") is MODEL_SPECS["garment-measurements"]


@pytest.mark.fast
def test_catalog_entries_are_immutable() -> None:
    with pytest.raises(TypeError):
        MODEL_SPECS["new"] = MODEL_SPECS["smpl"]
    with pytest.raises(TypeError):
        ASSET_SPECS["new"] = ASSET_SPECS["soma"]
