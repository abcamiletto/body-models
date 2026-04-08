"""Interface contract tests for all model classes/backends."""

from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

ASSET_DIR = Path(__file__).parent / "assets"
MODELS = ["smpl", "smplx", "flame", "skel", "anny", "mhr"]
BACKENDS = ["torch", "numpy", "jax"]


def _get_model_file(model_name: str) -> Path:
    model_dir = ASSET_DIR / model_name / "model"
    if not model_dir.exists():
        return model_dir
    for ext in (".npz", ".pkl"):
        for f in model_dir.glob(f"*{ext}"):
            return f
    return model_dir


def _class_name(model_name: str) -> str:
    return model_name.upper() if model_name != "flame" else "FLAME"


def _build_model(model_name: str, backend: str) -> Any:
    if backend == "jax":
        pytest.importorskip("jax")
        pytest.importorskip("flax")

    module = import_module(f"body_models.{model_name}.{backend}")
    cls = getattr(module, _class_name(model_name))
    model_path = _get_model_file(model_name)

    kwargs: dict[str, Any] = {}
    if model_name == "skel":
        if not model_path.exists():
            pytest.skip(f"Model assets not found: {model_path}")
        kwargs["gender"] = "male"
        kwargs["model_path"] = model_path
    elif model_name in {"smpl", "smplx", "flame"}:
        if not model_path.exists():
            pytest.skip(f"Model assets not found: {model_path}")
        kwargs["model_path"] = model_path
    elif model_path.exists():
        kwargs["model_path"] = model_path

    return cls(**kwargs)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model_name", MODELS)
def test_model_interface_attributes(model_name: str, backend: str) -> None:
    model = _build_model(model_name, backend)

    assert isinstance(model.num_joints, int)
    assert model.num_joints > 0

    assert isinstance(model.num_vertices, int)
    assert model.num_vertices > 0

    assert isinstance(model.joint_names, list)
    assert len(model.joint_names) == model.num_joints
    assert all(isinstance(name, str) for name in model.joint_names)

    params = model.get_rest_pose(batch_size=1)
    skeleton = model.forward_skeleton(**params)
    assert skeleton.ndim == 4
    assert skeleton.shape[-2:] == (4, 4)
    assert skeleton.shape[-3] == len(model.joint_names)
    assert skeleton.shape[-3] == model.num_joints

    assert model.faces.ndim == 2
    assert model.faces.shape[0] > 0

    assert model.rest_vertices.ndim == 2
    assert model.rest_vertices.shape[0] == model.num_vertices
    assert model.rest_vertices.shape[1] == 3
