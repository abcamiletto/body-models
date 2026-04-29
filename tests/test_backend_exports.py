"""Regression tests for backend module public exports."""

from importlib import import_module

import pytest


EXPECTED_PUBLIC_NAMES = {
    "body_models.anny.torch": ["ANNY"],
    "body_models.anny.numpy": ["ANNY"],
    "body_models.anny.jax": ["ANNY"],
    "body_models.flame.torch": ["FLAME"],
    "body_models.flame.numpy": ["FLAME"],
    "body_models.flame.jax": ["FLAME"],
    "body_models.brainco.torch": ["BrainCoHand"],
    "body_models.brainco.numpy": ["BrainCoHand"],
    "body_models.brainco.jax": ["BrainCoHand"],
    "body_models.garment_measurements.torch": ["GarmentMeasurements"],
    "body_models.garment_measurements.numpy": ["GarmentMeasurements"],
    "body_models.garment_measurements.jax": ["GarmentMeasurements"],
    "body_models.g1.torch": ["G1"],
    "body_models.g1.numpy": ["G1"],
    "body_models.g1.jax": ["G1"],
    "body_models.mhr.torch": ["MHR"],
    "body_models.mhr.numpy": ["MHR"],
    "body_models.mhr.jax": ["MHR"],
    "body_models.myofullbody.torch": ["MyoFullBody"],
    "body_models.myofullbody.numpy": ["MyoFullBody"],
    "body_models.myofullbody.jax": ["MyoFullBody"],
    "body_models.skel.torch": ["SKEL"],
    "body_models.skel.numpy": ["SKEL"],
    "body_models.skel.jax": ["SKEL"],
    "body_models.smpl.torch": ["SMPL"],
    "body_models.smpl.numpy": ["SMPL"],
    "body_models.smpl.jax": ["SMPL"],
    "body_models.smplh.torch": ["SMPLH"],
    "body_models.smplh.numpy": ["SMPLH"],
    "body_models.smplh.jax": ["SMPLH"],
    "body_models.mano.torch": ["MANO"],
    "body_models.mano.numpy": ["MANO"],
    "body_models.mano.jax": ["MANO"],
    "body_models.smplx.torch": ["SMPLX"],
    "body_models.smplx.numpy": ["SMPLX"],
    "body_models.smplx.jax": ["SMPLX"],
    "body_models.soma.torch": ["SOMA"],
    "body_models.soma.numpy": ["SOMA"],
    "body_models.soma.jax": ["SOMA"],
}


@pytest.mark.parametrize("module_name, expected_names", EXPECTED_PUBLIC_NAMES.items())
def test_backend_modules_define_expected_exports(module_name: str, expected_names: list[str]) -> None:
    if module_name.endswith(".torch"):
        pytest.importorskip("torch")
    if module_name.endswith(".jax"):
        pytest.importorskip("jax")
        pytest.importorskip("flax")

    module = import_module(module_name)
    assert sorted(module.__all__) == expected_names
