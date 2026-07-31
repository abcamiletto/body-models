"""Public model and asset catalog contracts."""

from importlib import import_module

import pytest

from body_models import ArticulatedModel
from body_models._catalog import ASSET_SPECS, DOWNLOAD_SPECS, MODEL_SPECS

MODEL_TARGETS = sorted({(spec.module, spec.class_name) for spec in MODEL_SPECS.values()})


@pytest.mark.fast
@pytest.mark.parametrize(("module_name", "class_name"), MODEL_TARGETS)
def test_catalog_models_are_exported_and_articulated(module_name, class_name) -> None:
    model_class = getattr(import_module(module_name), class_name)
    assert issubclass(model_class, ArticulatedModel)


@pytest.mark.fast
def test_catalog_aliases_share_one_model_class() -> None:
    aliases = ("humenv", "phc", "smpl-humanoid", "smplsim")
    classes = {getattr(import_module(MODEL_SPECS[name].module), MODEL_SPECS[name].class_name) for name in aliases}
    assert len(classes) == 1


@pytest.mark.fast
def test_download_catalog_is_importable_and_covers_families() -> None:
    assert all(
        any(asset == family or asset.startswith(f"{family}-") for family in DOWNLOAD_SPECS) for asset in ASSET_SPECS
    )
    for spec in DOWNLOAD_SPECS.values():
        assert callable(getattr(import_module(spec.module), spec.function))
