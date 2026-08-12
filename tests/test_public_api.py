"""Snapshot of the stable public API surface."""

from importlib import import_module

import model_cases
import pytest

import body_models
from body_models import _catalog as catalog

EXPECTED_ROOT_EXPORTS = (
    "ArrayRuntime",
    "CorrectiveBasis",
    "DenseCorrectiveBasis",
    "Joint",
    "KernelBackend",
    "LinearIdentity",
    "ParameterRole",
    "ParameterSpec",
    "PointRegressor",
    "RotationType",
    "RuntimeName",
    "SkinnedModel",
    "SkinningIdentity",
    "SkinningPose",
    "SkinningSpec",
    "SparseCorrectiveBasis",
    "SparseMatrix",
    "create_model",
    "list_models",
)

EXPECTED_PARAMETER_SPECS = {
    "anny": {
        "shape": ((6,), "identity", None),
        "body_pose": ((64, 3), "pose", "axis_angle"),
        "head_pose": ((60, 3), "pose", "axis_angle"),
        "hand_pose": ((38, 3), "pose", "axis_angle"),
        "global_rotation": ((3,), "transform", "axis_angle"),
        "global_translation": ((3,), "transform", None),
    },
    "flame": {
        "shape": ((300,), "identity", None),
        "expression": ((100,), "identity", None),
        "head_pose": ((4, 3), "pose", "axis_angle"),
        "head_rotation": ((3,), "pose", "axis_angle"),
        "global_rotation": ((3,), "transform", "axis_angle"),
        "global_translation": ((3,), "transform", None),
    },
    "garment_measurements": {
        "shape": ((15,), "identity", None),
        "body_pose": ((25, 3), "pose", "axis_angle"),
        "head_pose": ((3, 3), "pose", "axis_angle"),
        "hand_pose": ((30, 3), "pose", "axis_angle"),
        "pelvis_rotation": ((3,), "pose", "axis_angle"),
        "global_rotation": ((3,), "transform", "axis_angle"),
        "global_translation": ((3,), "transform", None),
    },
    "mano": {
        "shape": ((10,), "identity", None),
        "hand_pose": ((15, 3), "pose", "axis_angle"),
        "wrist_rotation": ((3,), "pose", "axis_angle"),
        "global_rotation": ((3,), "transform", "axis_angle"),
        "global_translation": ((3,), "transform", None),
    },
    "mhr": {
        "shape": ((45,), "identity", None),
        "expression": ((72,), "identity", None),
        "body_pose": ((94,), "pose", None),
        "head_pose": ((6,), "pose", None),
        "hand_pose": ((104,), "pose", None),
        "global_rotation": ((3,), "transform", "axis_angle"),
        "global_translation": ((3,), "transform", None),
    },
    "skel": {
        "shape": ((10,), "identity", None),
        "body_pose": ((43,), "pose", None),
        "head_pose": ((3,), "pose", None),
        "global_rotation": ((3,), "transform", "axis_angle"),
        "global_translation": ((3,), "transform", None),
    },
    "smpl": {
        "shape": ((10,), "identity", None),
        "body_pose": ((23, 3), "pose", "axis_angle"),
        "pelvis_rotation": ((3,), "pose", "axis_angle"),
        "global_rotation": ((3,), "transform", "axis_angle"),
        "global_translation": ((3,), "transform", None),
    },
    "smplh": {
        "shape": ((16,), "identity", None),
        "body_pose": ((21, 3), "pose", "axis_angle"),
        "hand_pose": ((30, 3), "pose", "axis_angle"),
        "pelvis_rotation": ((3,), "pose", "axis_angle"),
        "global_rotation": ((3,), "transform", "axis_angle"),
        "global_translation": ((3,), "transform", None),
    },
    "smplx": {
        "shape": ((300,), "identity", None),
        "expression": ((10,), "identity", None),
        "body_pose": ((21, 3), "pose", "axis_angle"),
        "head_pose": ((3, 3), "pose", "axis_angle"),
        "hand_pose": ((30, 3), "pose", "axis_angle"),
        "pelvis_rotation": ((3,), "pose", "axis_angle"),
        "global_rotation": ((3,), "transform", "axis_angle"),
        "global_translation": ((3,), "transform", None),
    },
    "soma": {
        "shape": ((128,), "identity", None),
        "body_pose": ((23, 3), "pose", "axis_angle"),
        "head_pose": ((5, 3), "pose", "axis_angle"),
        "hand_pose": ((48, 3), "pose", "axis_angle"),
        "global_rotation": ((3,), "transform", "axis_angle"),
        "global_translation": ((3,), "transform", None),
    },
}


@pytest.mark.fast
def test_root_exports_match_snapshot() -> None:
    assert sorted(body_models.__all__) == sorted(set(body_models.__all__))
    assert sorted(body_models.__all__) == list(EXPECTED_ROOT_EXPORTS)
    for name in body_models.__all__:
        assert getattr(body_models, name) is not None


@pytest.mark.fast
@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_backend_modules_export_model_class(backend) -> None:
    if backend != "numpy":
        pytest.importorskip(backend)
    for spec in catalog.MODEL_SPECS.values():
        module = import_module(f"{spec.module}.{backend}")
        assert spec.class_name in module.__all__


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_parameter_spec_matches_snapshot(name, model_class, kwargs) -> None:
    model = model_class(**kwargs)
    actual = {key: (tuple(spec.shape), spec.role, spec.rotation_type) for key, spec in model.parameter_spec.items()}
    expected = EXPECTED_PARAMETER_SPECS[name]

    assert list(actual) == list(expected)
    assert actual == expected
