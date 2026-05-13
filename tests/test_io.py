"""Model loading smoke tests."""

from pathlib import Path

import pytest

from body_models.anny.numpy import ANNY
from body_models.brainco.numpy import BrainCoHand
from body_models.flame.numpy import FLAME
from body_models.g1.numpy import G1
from body_models.garment_measurements.numpy import GarmentMeasurements
from body_models.mano.numpy import MANO
from body_models.mhr.numpy import MHR
from body_models.myofullbody.numpy import MyoFullBody
from body_models.skel.numpy import SKEL
from body_models.smpl.numpy import SMPL
from body_models.smplh.numpy import SMPLH
from body_models.smplx.numpy import SMPLX
from body_models.soma.numpy import SOMA

ASSETS = Path(__file__).parent / "assets"

MODEL_CASES = [
    (ANNY, ASSETS / "anny/model", {}),
    (BrainCoHand, ASSETS / "brainco/model", {"rotation_type": "hinge"}),
    (FLAME, ASSETS / "flame/model/FLAME_NEUTRAL.pkl", {}),
    (G1, ASSETS / "g1/model", {}),
    (GarmentMeasurements, ASSETS / "garment_measurements/model/garment_measurements.npz", {}),
    (MANO, ASSETS / "mano/model/right/MANO_RIGHT.pkl", {}),
    (MHR, ASSETS / "mhr/model", {}),
    (MyoFullBody, ASSETS / "myofullbody/model", {}),
    (SKEL, ASSETS / "skel/model/skel_male.pkl", {"gender": "male"}),
    (SMPL, ASSETS / "smpl/model/SMPL_NEUTRAL.npz", {}),
    (SMPLH, ASSETS / "smplh/model/neutral/model.npz", {}),
    (SMPLX, ASSETS / "smplx/model/SMPLX_NEUTRAL.npz", {}),
    (SOMA, ASSETS / "soma/model", {}),
]


@pytest.mark.parametrize(("model_class", "model_path", "kwargs"), MODEL_CASES)
def test_model_loads(model_class, model_path: Path, kwargs: dict) -> None:
    if not model_path.exists():
        pytest.skip(f"Missing model asset: {model_path}")

    model_class(model_path=model_path, **kwargs)
