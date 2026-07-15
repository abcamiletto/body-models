"""Shared model list for cross-model tests."""

from inspect import signature
from pathlib import Path

from body_models.anny import jax as anny_jax
from body_models.anny import numpy as anny_numpy
from body_models.anny import torch as anny_torch
from body_models.brainco import jax as brainco_jax
from body_models.brainco import numpy as brainco_numpy
from body_models.brainco import torch as brainco_torch
from body_models.flame import jax as flame_jax
from body_models.flame import numpy as flame_numpy
from body_models.flame import torch as flame_torch
from body_models.g1 import jax as g1_jax
from body_models.g1 import numpy as g1_numpy
from body_models.g1 import torch as g1_torch
from body_models.garment_measurements import jax as garment_jax
from body_models.garment_measurements import numpy as garment_numpy
from body_models.garment_measurements import torch as garment_torch
from body_models.mano import jax as mano_jax
from body_models.mano import numpy as mano_numpy
from body_models.mano import torch as mano_torch
from body_models.mhr import jax as mhr_jax
from body_models.mhr import numpy as mhr_numpy
from body_models.mhr import torch as mhr_torch
from body_models.myofullbody import jax as myofullbody_jax
from body_models.myofullbody import numpy as myofullbody_numpy
from body_models.myofullbody import torch as myofullbody_torch
from body_models.skel import jax as skel_jax
from body_models.skel import numpy as skel_numpy
from body_models.skel import torch as skel_torch
from body_models.smpl_humanoid import jax as smpl_humanoid_jax
from body_models.smpl_humanoid import numpy as smpl_humanoid_numpy
from body_models.smpl_humanoid import torch as smpl_humanoid_torch
from body_models.smpl import jax as smpl_jax
from body_models.smpl import numpy as smpl_numpy
from body_models.smpl import torch as smpl_torch
from body_models.smplh import jax as smplh_jax
from body_models.smplh import numpy as smplh_numpy
from body_models.smplh import torch as smplh_torch
from body_models.smplx import jax as smplx_jax
from body_models.smplx import numpy as smplx_numpy
from body_models.smplx import torch as smplx_torch
from body_models.soma import jax as soma_jax
from body_models.soma import numpy as soma_numpy
from body_models.soma import torch as soma_torch
from body_models.base import RigidBodyModel, SkinnedModel

ASSETS = Path(__file__).parent / "assets"

MODELS = [
    ("anny", anny_numpy.ANNY, anny_torch.ANNY, anny_jax.ANNY, {}),
    (
        "brainco",
        brainco_numpy.BrainCoHand,
        brainco_torch.BrainCoHand,
        brainco_jax.BrainCoHand,
        {},
    ),
    ("flame", flame_numpy.FLAME, flame_torch.FLAME, flame_jax.FLAME, {}),
    ("g1", g1_numpy.G1, g1_torch.G1, g1_jax.G1, {}),
    (
        "garment_measurements",
        garment_numpy.GarmentMeasurements,
        garment_torch.GarmentMeasurements,
        garment_jax.GarmentMeasurements,
        {},
    ),
    ("mano", mano_numpy.MANO, mano_torch.MANO, mano_jax.MANO, {"side": "right"}),
    ("mhr", mhr_numpy.MHR, mhr_torch.MHR, mhr_jax.MHR, {}),
    (
        "myofullbody",
        myofullbody_numpy.MyoFullBody,
        myofullbody_torch.MyoFullBody,
        myofullbody_jax.MyoFullBody,
        {},
    ),
    ("skel", skel_numpy.SKEL, skel_torch.SKEL, skel_jax.SKEL, {"gender": "male"}),
    ("smpl", smpl_numpy.SMPL, smpl_torch.SMPL, smpl_jax.SMPL, {"gender": "neutral"}),
    (
        "smpl_humanoid",
        smpl_humanoid_numpy.SmplHumanoid,
        smpl_humanoid_torch.SmplHumanoid,
        smpl_humanoid_jax.SmplHumanoid,
        {},
    ),
    ("smplh", smplh_numpy.SMPLH, smplh_torch.SMPLH, smplh_jax.SMPLH, {"gender": "neutral"}),
    ("smplx", smplx_numpy.SMPLX, smplx_torch.SMPLX, smplx_jax.SMPLX, {"gender": "neutral"}),
    ("soma", soma_numpy.SOMA, soma_torch.SOMA, soma_jax.SOMA, {}),
]

RIGID_BODY_MODELS = [model for model in MODELS if issubclass(model[1], RigidBodyModel)]

SKINNED_MODELS = [model for model in MODELS if issubclass(model[1], SkinnedModel)]

REFERENCE_MODELS = [model for model in MODELS if (ASSETS / model[0] / "inputs" / "0.json").exists()]


def forward_skeleton(model, params, **kwargs):
    """Call a model-specific skeleton signature with the parameters it accepts."""
    arguments = dict(params) | kwargs
    accepted = signature(model.forward_skeleton).parameters
    return model.forward_skeleton(**{key: value for key, value in arguments.items() if key in accepted})
