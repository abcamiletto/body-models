"""Public model and asset catalog contracts."""

from importlib import import_module

import pytest

from body_models._catalog import ASSET_SPECS, DOWNLOAD_SPECS, MODEL_SPECS
from body_models._registry import get_model_spec

MODEL_TARGETS = sorted({(spec.module, spec.class_name) for spec in MODEL_SPECS.values()})


@pytest.mark.fast
@pytest.mark.parametrize(("module_name", "class_name"), MODEL_TARGETS)
def test_catalog_models_are_exported_from_their_packages(module_name, class_name) -> None:
    module = import_module(module_name)
    assert hasattr(module, class_name)


@pytest.mark.fast
def test_catalog_aliases_share_one_model_class() -> None:
    aliases = ("humenv", "phc", "smpl-humanoid", "smplsim")
    classes = {getattr(import_module(MODEL_SPECS[name].module), MODEL_SPECS[name].class_name) for name in aliases}
    assert len(classes) == 1


@pytest.mark.fast
def test_registry_normalizes_public_names() -> None:
    assert get_model_spec("smpl_humanoid") is MODEL_SPECS["smpl-humanoid"]
    assert get_model_spec(" GARMENT_MEASUREMENTS ") is MODEL_SPECS["garment-measurements"]


@pytest.mark.fast
def test_download_catalog_is_importable_and_covers_families() -> None:
    assert all(
        any(asset == family or asset.startswith(f"{family}-") for family in DOWNLOAD_SPECS) for asset in ASSET_SPECS
    )
    for spec in DOWNLOAD_SPECS.values():
        assert callable(getattr(import_module(spec.module), spec.function))
