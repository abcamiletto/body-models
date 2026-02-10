"""Lightweight smoke tests used for Python/dependency version compatibility in CI."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

ASSET_DIR = Path(__file__).parent / "assets"
MODELS = ("smpl", "smplx", "skel", "flame", "anny", "mhr")
BACKENDS = ("numpy", "torch", "jax")


def get_model_file(model_name: str) -> Path:
    """Get the test asset path for a given model."""
    model_dir = ASSET_DIR / model_name / "model"
    if not model_dir.exists():
        return model_dir

    for ext in (".npz", ".pkl"):
        for file_path in model_dir.glob(f"*{ext}"):
            return file_path
    return model_dir


def get_model(backend: str, model_name: str, model_path: Path) -> Any:
    """Instantiate a model for a specific backend."""
    if backend == "numpy":
        if model_name == "smpl":
            from body_models.smpl.numpy import SMPL

            return SMPL(model_path=model_path)
        if model_name == "smplx":
            from body_models.smplx.numpy import SMPLX

            return SMPLX(model_path=model_path)
        if model_name == "skel":
            from body_models.skel.numpy import SKEL

            return SKEL(gender="male", model_path=model_path)
        if model_name == "flame":
            from body_models.flame.numpy import FLAME

            return FLAME(model_path=model_path)
        if model_name == "anny":
            from body_models.anny.numpy import ANNY

            return ANNY(model_path=model_path)
        if model_name == "mhr":
            from body_models.mhr.numpy import MHR

            return MHR(model_path=model_path)

    if backend == "torch":
        if model_name == "smpl":
            from body_models.smpl.torch import SMPL

            return SMPL(model_path=model_path)
        if model_name == "smplx":
            from body_models.smplx.torch import SMPLX

            return SMPLX(model_path=model_path)
        if model_name == "skel":
            from body_models.skel.torch import SKEL

            return SKEL(gender="male", model_path=model_path)
        if model_name == "flame":
            from body_models.flame.torch import FLAME

            return FLAME(model_path=model_path)
        if model_name == "anny":
            from body_models.anny.torch import ANNY

            return ANNY(model_path=model_path)
        if model_name == "mhr":
            from body_models.mhr.torch import MHR

            return MHR(model_path=model_path)

    if backend == "jax":
        if model_name == "smpl":
            from body_models.smpl.jax import SMPL

            return SMPL(model_path=model_path)
        if model_name == "smplx":
            from body_models.smplx.jax import SMPLX

            return SMPLX(model_path=model_path)
        if model_name == "skel":
            from body_models.skel.jax import SKEL

            return SKEL(gender="male", model_path=model_path)
        if model_name == "flame":
            from body_models.flame.jax import FLAME

            return FLAME(model_path=model_path)
        if model_name == "anny":
            from body_models.anny.jax import ANNY

            return ANNY(model_path=model_path)
        if model_name == "mhr":
            from body_models.mhr.jax import MHR

            return MHR(model_path=model_path)

    raise ValueError(f"Unsupported backend/model combination: {backend}/{model_name}")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model_name", MODELS)
def test_forward_smoke(backend: str, model_name: str) -> None:
    """Ensure basic forward passes work across all backends and models."""
    if backend == "torch":
        pytest.importorskip("torch")
    if backend == "jax":
        pytest.importorskip("jax")

    model_path = get_model_file(model_name)
    if not model_path.exists():
        pytest.skip(f"Model assets not found: {model_path}")

    model = get_model(backend, model_name, model_path)
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
