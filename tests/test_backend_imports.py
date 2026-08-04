"""Backend-specific public model imports."""

import importlib
import inspect
import pickle

import pytest

from body_models import SkinnedModel
from body_models import _catalog as catalog


@pytest.mark.fast
@pytest.mark.parametrize("spec", catalog.MODEL_SPECS.values())
@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_backend_model_signature(spec, backend) -> None:
    package = spec.module
    model_class = getattr(importlib.import_module(f"{package}.{backend}"), spec.class_name)
    base_class = getattr(importlib.import_module(package), spec.class_name)
    parameters = inspect.signature(model_class).parameters

    assert issubclass(model_class, base_class)
    assert model_class.__module__ == f"{package}.{backend}"
    assert "runtime" not in parameters
    has_skinning_backend = backend == "torch" and issubclass(base_class, SkinnedModel)
    assert ("skinning_backend" in parameters) is has_skinning_backend


@pytest.mark.fast
@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_backend_model_binds_runtime(backend) -> None:
    if backend != "numpy":
        pytest.importorskip(backend)
    model_class = importlib.import_module(f"body_models.g1.{backend}").G1

    model = model_class()

    assert model.runtime.name == backend
    if backend == "torch":
        import torch

        assert isinstance(model, torch.nn.Module)
    assert type(pickle.loads(pickle.dumps(model))) is model_class
    with pytest.raises(TypeError, match="unexpected keyword argument 'runtime'"):
        model_class(runtime="numpy")


@pytest.mark.fast
def test_torch_model_exposes_skinning_backend() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("warp")
    from body_models.smpl.torch import SMPL

    model = SMPL(gender="neutral", skinning_backend="warp")

    assert model.runtime.skinning_backend == "warp"
