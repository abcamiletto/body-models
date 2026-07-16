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


@dataclass(frozen=True)
class CredentialSpec:
    """Account metadata for a licensed model download."""

    account: str
    url: str


@dataclass(frozen=True)
class DownloadSpec:
    """Lazy downloader and output contract for one model family."""

    module: str
    function: str
    output_key: str | None = None
    credentials: CredentialSpec | None = None


def _model(module: str, class_name: str, kind: ModelKind, **defaults: Any) -> ModelSpec:
    return ModelSpec(module, class_name, kind, MappingProxyType(defaults))


MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "anny": _model("body_models.anny", "ANNY", "skinned"),
        "brainco": _model("body_models.brainco", "BrainCoHand", "rigid"),
        "flame": _model("body_models.flame", "FLAME", "skinned"),
        "g1": _model("body_models.g1", "G1", "rigid"),
        "garment-measurements": _model(
            "body_models.garment_measurements",
            "GarmentMeasurements",
            "skinned",
        ),
        "humenv": _model("body_models.smpl_humanoid", "SmplHumanoid", "rigid", source="humenv"),
        "mano": _model("body_models.mano", "MANO", "skinned", side="right"),
        "mhr": _model("body_models.mhr", "MHR", "skinned"),
        "myofullbody": _model("body_models.myofullbody", "MyoFullBody", "rigid"),
        "phc": _model("body_models.smpl_humanoid", "SmplHumanoid", "rigid", source="phc"),
        "skel": _model("body_models.skel", "SKEL", "skinned", gender="male"),
        "smpl": _model("body_models.smpl", "SMPL", "skinned", gender="neutral"),
        "smpl-humanoid": _model("body_models.smpl_humanoid", "SmplHumanoid", "rigid"),
        "smplh": _model("body_models.smplh", "SMPLH", "skinned", gender="neutral"),
        "smplsim": _model("body_models.smpl_humanoid", "SmplHumanoid", "rigid", source="smplsim"),
        "smplx": _model("body_models.smplx", "SMPLX", "skinned", gender="neutral"),
        "soma": _model("body_models.soma", "SOMA", "skinned"),
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


def _credentials(account: str, url: str) -> CredentialSpec:
    return CredentialSpec(account, url)


DOWNLOAD_SPECS: Mapping[str, DownloadSpec] = MappingProxyType(
    {
        "smpl": DownloadSpec(
            "body_models.download",
            "download_smpl",
            credentials=_credentials("SMPL", "https://smpl.is.tue.mpg.de/"),
        ),
        "smplh": DownloadSpec(
            "body_models.download",
            "download_smplh",
            credentials=_credentials("SMPLH", "https://mano.is.tue.mpg.de/"),
        ),
        "mano": DownloadSpec(
            "body_models.download",
            "download_mano",
            credentials=_credentials("MANO", "https://mano.is.tue.mpg.de/"),
        ),
        "smplx": DownloadSpec(
            "body_models.download",
            "download_smplx",
            credentials=_credentials("SMPLX", "https://smpl-x.is.tue.mpg.de/"),
        ),
        "smpl-humanoid": DownloadSpec("body_models.robots.smpl_humanoid.io", "download_assets"),
        "skel": DownloadSpec(
            "body_models.download",
            "download_skel_assets",
            credentials=_credentials("SKEL", "https://skel.is.tue.mpg.de/"),
        ),
        "flame": DownloadSpec(
            "body_models.download",
            "download_flame",
            output_key="flame",
            credentials=_credentials("FLAME", "https://flame.is.tue.mpg.de/"),
        ),
        "anny": DownloadSpec("body_models.bodies.anny.io", "download_model", output_key="anny"),
        "brainco": DownloadSpec("body_models.robots.brainco.io", "download_model", output_key="brainco"),
        "mhr": DownloadSpec("body_models.bodies.mhr.io", "download_model", output_key="mhr"),
        "g1": DownloadSpec("body_models.robots.g1.io", "download_model", output_key="g1"),
        "soma": DownloadSpec("body_models.bodies.soma.io", "download_model", output_key="soma"),
        "garment-measurements": DownloadSpec(
            "body_models.bodies.garment_measurements.io",
            "download_model",
            output_key="garment-measurements",
        ),
        "myofullbody": DownloadSpec(
            "body_models.skeletons.myofullbody.io",
            "download_model",
            output_key="myofullbody",
        ),
    }
)


__all__ = [
    "ASSET_SPECS",
    "DOWNLOAD_SPECS",
    "MODEL_SPECS",
    "AssetSpec",
    "CredentialSpec",
    "DownloadSpec",
    "ModelKind",
    "ModelSpec",
]
