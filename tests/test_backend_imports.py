"""Backend-specific public model imports."""

import importlib
import inspect

import pytest

from body_models import SkinnedModel
from body_models import _catalog as catalog


@pytest.mark.fast
def test_backend_model_signatures() -> None:
    for spec in catalog.MODEL_SPECS.values():
        package = spec.module
        base_class = getattr(importlib.import_module(package), spec.class_name)
        for backend in ("numpy", "torch", "jax"):
            model_class = getattr(importlib.import_module(f"{package}.{backend}"), spec.class_name)
            parameters = inspect.signature(model_class).parameters

            assert issubclass(model_class, base_class)
            assert model_class.__module__ == f"{package}.{backend}"
            assert "runtime" not in parameters
            has_skinning_backend = backend == "torch" and issubclass(base_class, SkinnedModel)
            assert ("skinning_backend" in parameters) is has_skinning_backend
