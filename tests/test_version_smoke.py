"""Lightweight smoke tests used for Python/dependency version compatibility in CI."""

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import model_assets

MODEL_CASES = (
    pytest.param("smpl", {}, id="smpl"),
    pytest.param("smplh", {}, id="smplh"),
    pytest.param("mano", {}, id="mano"),
    pytest.param("smplx", {}, id="smplx"),
    pytest.param("skel", {}, id="skel"),
    pytest.param("flame", {}, id="flame"),
    pytest.param("anny", {}, id="anny"),
    pytest.param("mhr", {}, id="mhr"),
    pytest.param("soma", {"model_type": "soma"}, id="soma"),
    pytest.param("soma", {"model_type": "anny"}, id="soma-anny"),
    pytest.param("soma", {"model_type": "mhr"}, id="soma-mhr"),
    pytest.param("soma", {"model_type": "smpl"}, id="soma-smpl"),
    pytest.param("soma", {"model_type": "smplx"}, id="soma-smplx"),
    pytest.param("garment_measurements", {}, id="garment-measurements"),
)
BACKENDS = ("numpy", "torch", "jax")


def get_model_file(model_name: str) -> Path:
    """Get the test asset path for a given model."""
    return model_assets.get_model_file(model_name)


def get_required_model_files(model_name: str, model_kwargs: dict[str, str]) -> list[Path]:
    paths = [get_model_file(model_name)]
    if model_name != "soma":
        return paths

    nested_model_type = model_kwargs.get("model_type")
    if nested_model_type in model_assets.SOMA_NESTED_MODEL_TYPES:
        paths.append(model_assets.get_model_file(nested_model_type))
    return paths


def get_model(backend: str, model_name: str, model_path: Path, **kwargs) -> Any:
    """Instantiate a model for a specific backend."""
    module = import_module(f"body_models.{model_name}.{backend}")
    model_class = getattr(module, model_assets.CLASS_NAMES[model_name])
    ctor_kwargs = {"model_path": model_path, **kwargs}
    if model_name == "skel":
        ctor_kwargs["gender"] = "male"
    return model_class(**ctor_kwargs)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("model_name", "model_kwargs"), MODEL_CASES)
def test_forward_smoke(backend: str, model_name: str, model_kwargs: dict[str, str]) -> None:
    """Ensure basic forward passes work across all backends and models."""
    if backend == "torch":
        pytest.importorskip("torch")
    if backend == "jax":
        pytest.importorskip("jax")
        pytest.importorskip("flax")

    required_paths = get_required_model_files(model_name, model_kwargs)
    missing_path = next((path for path in required_paths if not path.exists()), None)
    if missing_path is not None:
        pytest.skip(f"Model assets not found: {missing_path}")

    model_path = required_paths[0]
    model = get_model(backend, model_name, model_path, **model_kwargs)
    params = model.get_rest_pose(batch_size=1)

    if backend == "torch":
        import torch

        with torch.no_grad():
            vertices = model.forward_vertices(**params)
            skeleton = model.forward_skeleton(**params)
    else:
        vertices = model.forward_vertices(**params)
        skeleton = model.forward_skeleton(**params)

    vertices_np = np.asarray(vertices)
    skeleton_np = np.asarray(skeleton)

    assert vertices_np.shape[0] == 1
    assert vertices_np.shape[-1] == 3
    assert skeleton_np.shape[0] == 1
    assert np.isfinite(vertices_np).all()
    assert np.isfinite(skeleton_np).all()
