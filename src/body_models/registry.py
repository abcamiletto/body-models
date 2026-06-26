"""Model factory helpers."""

from __future__ import annotations

from fnmatch import fnmatchcase
from importlib import import_module
from typing import Any, Literal

from body_models.base import RigidBodyModel, SkinnedModel

Backend = Literal["numpy", "torch", "jax"]
ModelInfo = tuple[str, str, dict[str, Any]]

_MODELS: dict[str, ModelInfo] = {
    "anny": ("body_models.bodies.anny", "ANNY", {}),
    "brainco": ("body_models.robots.brainco", "BrainCoHand", {}),
    "flame": ("body_models.parts.flame", "FLAME", {}),
    "g1": ("body_models.robots.g1", "G1", {}),
    "garment_measurements": ("body_models.bodies.garment_measurements", "GarmentMeasurements", {}),
    "mano": ("body_models.parts.mano", "MANO", {"side": "right"}),
    "mhr": ("body_models.bodies.mhr", "MHR", {}),
    "myofullbody": ("body_models.skeletons.myofullbody", "MyoFullBody", {}),
    "skel": ("body_models.skeletons.skel", "SKEL", {"gender": "male"}),
    "smpl": ("body_models.bodies.smpl", "SMPL", {"gender": "neutral"}),
    "smpl_humanoid": ("body_models.robots.smpl_humanoid", "SmplHumanoid", {}),
    "meta_motivo": ("body_models.robots.smpl_humanoid", "SmplHumanoid", {"model": "meta_motivo"}),
    "phc": ("body_models.robots.smpl_humanoid", "SmplHumanoid", {"model": "phc"}),
    "smplsim": ("body_models.robots.smpl_humanoid", "SmplHumanoid", {"model": "smplsim"}),
    "smplh": ("body_models.bodies.smplh", "SMPLH", {"gender": "neutral"}),
    "smplx": ("body_models.bodies.smplx", "SMPLX", {"gender": "neutral"}),
    "soma": ("body_models.bodies.soma", "SOMA", {}),
}


def create_model(
    model_name: str,
    *,
    backend: Backend = "numpy",
    pretrained: bool = False,
    **kwargs: Any,
) -> SkinnedModel | RigidBodyModel:
    """Create a body model by name."""
    name = model_name.strip().lower().replace("-", "_")
    try:
        module_name, class_name, defaults = _MODELS[name]
    except KeyError as exc:
        available = ", ".join(list_models())
        raise ValueError(f"Unknown model {model_name!r}. Available models: {available}") from exc

    module = import_module(f"{module_name}.{backend}")
    model_cls = getattr(module, class_name)
    model_kwargs = defaults | kwargs
    return model_cls(**model_kwargs)


def list_models(filter: str = "") -> list[str]:
    """List registered model names."""
    names = sorted(_MODELS)
    if not filter:
        return names
    return [name for name in names if fnmatchcase(name, filter)]


__all__ = ["Backend", "create_model", "list_models"]
