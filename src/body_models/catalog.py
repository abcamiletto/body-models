"""Authoritative catalog of public models and configurable assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

ModelKind = Literal["skinned", "rigid"]


@dataclass(frozen=True)
class ModelSpec:
    """Lazy import and constructor defaults for one public factory name."""

    module: str
    class_name: str
    kind: ModelKind
    defaults: Mapping[str, Any] = field(default_factory=dict)

    @property
    def public_module(self) -> str:
        return self.module.removeprefix("body_models.")


@dataclass(frozen=True)
class AssetSpec:
    """Validation route for one persistent asset configuration key."""

    validation_module: str


def _model(module: str, class_name: str, kind: ModelKind, **defaults: Any) -> ModelSpec:
    return ModelSpec(module, class_name, kind, MappingProxyType(defaults))


MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "anny": _model("body_models.bodies.anny", "ANNY", "skinned"),
        "brainco": _model("body_models.robots.brainco", "BrainCoHand", "rigid"),
        "flame": _model("body_models.parts.flame", "FLAME", "skinned"),
        "g1": _model("body_models.robots.g1", "G1", "rigid"),
        "garment-measurements": _model(
            "body_models.bodies.garment_measurements",
            "GarmentMeasurements",
            "skinned",
        ),
        "humenv": _model("body_models.robots.smpl_humanoid", "SmplHumanoid", "rigid", source="humenv"),
        "mano": _model("body_models.parts.mano", "MANO", "skinned", side="right"),
        "mhr": _model("body_models.bodies.mhr", "MHR", "skinned"),
        "myofullbody": _model("body_models.skeletons.myofullbody", "MyoFullBody", "rigid"),
        "phc": _model("body_models.robots.smpl_humanoid", "SmplHumanoid", "rigid", source="phc"),
        "skel": _model("body_models.skeletons.skel", "SKEL", "skinned", gender="male"),
        "smpl": _model("body_models.bodies.smpl", "SMPL", "skinned", gender="neutral"),
        "smpl-humanoid": _model("body_models.robots.smpl_humanoid", "SmplHumanoid", "rigid"),
        "smplh": _model("body_models.bodies.smplh", "SMPLH", "skinned", gender="neutral"),
        "smplsim": _model("body_models.robots.smpl_humanoid", "SmplHumanoid", "rigid", source="smplsim"),
        "smplx": _model("body_models.bodies.smplx", "SMPLX", "skinned", gender="neutral"),
        "soma": _model("body_models.bodies.soma", "SOMA", "skinned"),
    }
)


def _assets(module: str, *names: str) -> dict[str, AssetSpec]:
    return {name: AssetSpec(module) for name in names}


ASSET_SPECS: Mapping[str, AssetSpec] = MappingProxyType(
    {
        **_assets("body_models.bodies.smpl.io", "smpl-male", "smpl-female", "smpl-neutral"),
        **_assets("body_models.bodies.smplx.io", "smplx-male", "smplx-female", "smplx-neutral"),
        **_assets("body_models.bodies.smplh.io", "smplh-male", "smplh-female", "smplh-neutral"),
        **_assets(
            "body_models.robots.smpl_humanoid.io",
            "smpl-humanoid-humenv",
            "smpl-humanoid-phc",
            "smpl-humanoid-smplsim",
        ),
        **_assets("body_models.parts.mano.io", "mano-right", "mano-left"),
        **_assets("body_models.skeletons.skel.io", "skel-male", "skel-female"),
        **_assets("body_models.bodies.anny.io", "anny"),
        **_assets("body_models.bodies.mhr.io", "mhr"),
        **_assets("body_models.parts.flame.io", "flame"),
        **_assets("body_models.robots.brainco.io", "brainco"),
        **_assets("body_models.robots.g1.io", "g1"),
        **_assets("body_models.bodies.soma.io", "soma"),
        **_assets("body_models.bodies.garment_measurements.io", "garment-measurements"),
        **_assets("body_models.skeletons.myofullbody.io", "myofullbody"),
    }
)


PUBLIC_MODULES: Mapping[str, str] = MappingProxyType(
    {spec.public_module.rsplit(".", 1)[-1]: spec.public_module for spec in MODEL_SPECS.values()}
)


__all__ = ["ASSET_SPECS", "MODEL_SPECS", "PUBLIC_MODULES", "AssetSpec", "ModelKind", "ModelSpec"]
