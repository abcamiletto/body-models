"""Shared model list for cross-model tests."""

from inspect import signature
from pathlib import Path

from body_models._base import RigidBodyModel, SkinnedModel
from body_models.anny import ANNY
from body_models.brainco import BrainCoHand
from body_models.flame import FLAME
from body_models.g1 import G1
from body_models.garment_measurements import GarmentMeasurements
from body_models.mano import MANO
from body_models.mhr import MHR
from body_models.myofullbody import MyoFullBody
from body_models.skel import SKEL
from body_models.smpl import SMPL
from body_models.smpl_humanoid import SmplHumanoid
from body_models.smplh import SMPLH
from body_models.smplx import SMPLX
from body_models.soma import SOMA

ASSETS = Path(__file__).parent / "assets"

MODELS = [
    ("anny", ANNY, {}),
    ("brainco", BrainCoHand, {}),
    ("flame", FLAME, {}),
    ("g1", G1, {}),
    ("garment_measurements", GarmentMeasurements, {}),
    ("mano", MANO, {"side": "right"}),
    ("mhr", MHR, {}),
    ("myofullbody", MyoFullBody, {}),
    ("skel", SKEL, {}),
    ("smpl", SMPL, {"gender": "neutral"}),
    ("smpl_humanoid", SmplHumanoid, {}),
    ("smplh", SMPLH, {"gender": "neutral"}),
    ("smplx", SMPLX, {"gender": "neutral"}),
    ("soma", SOMA, {}),
]

RIGID_BODY_MODELS = [model for model in MODELS if issubclass(model[1], RigidBodyModel)]

SKINNED_MODELS = [model for model in MODELS if issubclass(model[1], SkinnedModel)]

REFERENCE_MODELS = [model for model in MODELS if (ASSETS / model[0] / "inputs" / "0.json").exists()]


def forward_skeleton(model, params, **kwargs):
    """Call a model-specific skeleton signature with the parameters it accepts."""
    arguments = dict(params) | kwargs
    accepted = signature(model.forward_skeleton).parameters
    return model.forward_skeleton(**{key: value for key, value in arguments.items() if key in accepted})


def prepare_states(model, params):
    """Prepare model-specific identity and pose state from public parameters."""
    identity_params = {name: params[name] for name, spec in model.parameter_spec.items() if spec.role == "identity"}
    identity = model.prepare_identity(**identity_params)
    pose_params = {name: params[name] for name, spec in model.parameter_spec.items() if spec.role == "pose"}
    if "identity" in signature(model.prepare_pose).parameters:
        pose_params["identity"] = identity
    pose = model.prepare_pose(**pose_params)
    return identity, pose


def with_prepared_identity(model, params, identity):
    """Replace raw identity controls with prepared state in forward arguments."""
    raw_identity = {name for name, spec in model.parameter_spec.items() if spec.role == "identity"}
    return {key: value for key, value in params.items() if key not in raw_identity} | {"identity": identity}
