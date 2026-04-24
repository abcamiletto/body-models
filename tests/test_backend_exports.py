"""Regression tests for backend module public exports."""

from importlib import import_module

import pytest


EXPECTED_PUBLIC_NAMES = {
    "body_models.anny.torch": ["ANNY"],
    "body_models.anny.numpy": ["ANNY", "from_native_args", "to_native_outputs"],
    "body_models.anny.jax": ["ANNY", "from_native_args", "to_native_outputs"],
    "body_models.flame.torch": ["FLAME", "from_native_args", "to_native_outputs"],
    "body_models.flame.numpy": ["FLAME"],
    "body_models.flame.jax": ["FLAME"],
    "body_models.garment_measurements.torch": ["GarmentMeasurements"],
    "body_models.garment_measurements.numpy": ["GarmentMeasurements"],
    "body_models.garment_measurements.jax": ["GarmentMeasurements"],
    "body_models.g1.torch": ["G1"],
    "body_models.g1.numpy": ["G1"],
    "body_models.g1.jax": ["G1"],
    "body_models.mhr.torch": ["MHR"],
    "body_models.mhr.numpy": ["MHR"],
    "body_models.mhr.jax": ["MHR"],
    "body_models.skel.torch": ["SKEL", "from_native_args", "to_native_outputs"],
    "body_models.skel.numpy": ["SKEL", "from_native_args", "to_native_outputs"],
    "body_models.skel.jax": ["SKEL", "from_native_args", "to_native_outputs"],
    "body_models.smpl.torch": ["SMPL", "from_native_args", "to_native_outputs"],
    "body_models.smpl.numpy": ["SMPL"],
    "body_models.smpl.jax": ["SMPL"],
    "body_models.smplx.torch": ["SMPLX", "from_native_args", "to_native_outputs"],
    "body_models.smplx.numpy": ["SMPLX"],
    "body_models.smplx.jax": ["SMPLX"],
    "body_models.soma.torch": ["SOMA"],
    "body_models.soma.numpy": ["SOMA"],
    "body_models.soma.jax": ["SOMA"],
}


@pytest.mark.parametrize("module_name, expected_names", EXPECTED_PUBLIC_NAMES.items())
def test_backend_modules_only_export_expected_names(module_name: str, expected_names: list[str]) -> None:
    if module_name.endswith(".torch"):
        pytest.importorskip("torch")
    if module_name.endswith(".jax"):
        pytest.importorskip("jax")
        pytest.importorskip("flax")

    module = import_module(module_name)
    public_names = sorted(name for name in vars(module) if not name.startswith("_"))
    assert public_names == expected_names
