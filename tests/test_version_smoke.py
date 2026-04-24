"""Lightweight smoke tests used for Python/dependency version compatibility in CI."""

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ASSET_DIR = Path(__file__).parent / "assets"
MODEL_FILES = {
    "smpl": "SMPL_NEUTRAL.npz",
    "smplx": "SMPLX_NEUTRAL.npz",
    "flame": "FLAME_NEUTRAL.pkl",
    "skel": "skel_male.pkl",
}
CLASS_NAMES = {name: ("FLAME" if name == "flame" else name.upper()) for name in (*MODEL_FILES, "anny", "mhr", "soma")}
CLASS_NAMES["garment_measurements"] = "GarmentMeasurements"
MODEL_CASES = (
    pytest.param("smpl", {}, id="smpl"),
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
    if model_name == "garment_measurements":
        return ASSET_DIR / "garment_measurements" / "model" / "garment_measurements.npz"

    if model_name == "soma":
        from body_models.soma.io import get_model_path

        return get_model_path()

    model_dir = ASSET_DIR / model_name / "model"
    if not model_dir.exists():
        return model_dir

    filename = MODEL_FILES.get(model_name)
    return model_dir if filename is None else model_dir / filename


def get_required_model_files(model_name: str, model_kwargs: dict[str, str]) -> list[Path]:
    paths = [get_model_file(model_name)]
    if model_name != "soma":
        return paths

    nested_model_type = model_kwargs.get("model_type")
    if nested_model_type in MODEL_FILES:
        paths.append(ASSET_DIR / nested_model_type / "model" / MODEL_FILES[nested_model_type])
    return paths


def get_model(backend: str, model_name: str, model_path: Path, **kwargs) -> Any:
    """Instantiate a model for a specific backend."""
    module = import_module(f"body_models.{model_name}.{backend}")
    model_class = getattr(module, CLASS_NAMES[model_name])
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
    else:
        vertices = model.forward_vertices(**params)

    vertices_np = np.asarray(vertices)

    assert vertices_np.shape[0] == 1
    assert vertices_np.shape[-1] == 3
    assert np.isfinite(vertices_np).all()
    if hasattr(model, "forward_skeleton"):
        skeleton = model.forward_skeleton(**params)
        skeleton_np = np.asarray(skeleton)
        assert skeleton_np.shape[0] == 1
        assert np.isfinite(skeleton_np).all()
