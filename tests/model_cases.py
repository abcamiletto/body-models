"""Shared model list for cross-model tests."""

from importlib import import_module
from pathlib import Path

from body_models import SkinnedModel
from body_models.anny.numpy import ANNY
from body_models.flame.numpy import FLAME
from body_models.garment_measurements.numpy import GarmentMeasurements
from body_models.mano.numpy import MANO
from body_models.mhr.numpy import MHR
from body_models.skel.numpy import SKEL
from body_models.smpl.numpy import SMPL
from body_models.smplh.numpy import SMPLH
from body_models.smplx.numpy import SMPLX
from body_models.soma.numpy import SOMA

ASSETS = Path(__file__).parent / "assets"

MODELS = [
    ("anny", ANNY, {}),
    ("flame", FLAME, {}),
    ("garment_measurements", GarmentMeasurements, {}),
    ("mano", MANO, {"side": "right"}),
    ("mhr", MHR, {}),
    ("skel", SKEL, {}),
    ("smpl", SMPL, {"gender": "neutral"}),
    ("smplh", SMPLH, {"gender": "neutral"}),
    ("smplx", SMPLX, {"gender": "neutral"}),
    ("soma", SOMA, {}),
]

SKINNED_MODELS = [model for model in MODELS if issubclass(model[1], SkinnedModel)]

REFERENCE_MODELS = [model for model in MODELS if (ASSETS / model[0] / "inputs" / "0.json").exists()]


def backend_model_class(name: str, backend: str):
    """Return a model's public class for one array backend."""
    model_class = next(model_class for model_name, model_class, _ in MODELS if model_name == name)
    module = import_module(f"body_models.{name}.{backend}")
    return getattr(module, model_class.__name__)


def forward_skeleton(model, params, **kwargs):
    """Call a model skeleton with its complete public parameter mapping."""
    return model.forward_skeleton(**params, **kwargs)


def prepare_states(model, params):
    """Prepare model-specific identity and pose state from public parameters."""
    identity_params = {name: params[name] for name, spec in model.parameter_spec.items() if spec.role == "identity"}
    identity = model.prepare_identity(**identity_params)
    pose_params = {name: params[name] for name, spec in model.parameter_spec.items() if spec.role == "pose"}
    pose = model.prepare_pose(**pose_params, identity=identity)
    return identity, pose


def with_prepared_identity(model, params, identity):
    """Replace raw identity controls with prepared state in forward arguments."""
    raw_identity = {name for name, spec in model.parameter_spec.items() if spec.role == "identity"}
    return {key: value for key, value in params.items() if key not in raw_identity} | {"identity": identity}
