"""Public model and asset catalog contracts."""

from importlib import import_module
from typing import get_type_hints

import pytest

from body_models import ArticulatedModel
from body_models._catalog import ASSET_SPECS, DOWNLOAD_SPECS, MODEL_SPECS
from body_models._registry import get_model_spec

MODEL_TARGETS = sorted({(spec.module, spec.class_name) for spec in MODEL_SPECS.values()})


@pytest.mark.fast
@pytest.mark.parametrize(("module_name", "class_name"), MODEL_TARGETS)
def test_catalog_models_are_exported_and_articulated(module_name, class_name) -> None:
    model_class = getattr(import_module(module_name), class_name)
    assert issubclass(model_class, ArticulatedModel)


@pytest.mark.fast
@pytest.mark.parametrize(("module_name", "class_name"), MODEL_TARGETS)
def test_prepared_state_types_are_exported_from_model_packages(module_name, class_name) -> None:
    module = import_module(module_name)
    model_class = getattr(module, class_name)
    public_values = tuple(getattr(module, name) for name in module.__all__)
    for method_name in ("prepare_identity", "prepare_pose"):
        if hasattr(model_class, method_name):
            return_type = get_type_hints(getattr(model_class, method_name))["return"]
            assert return_type in public_values


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
