"""Authoritative catalog of public models and configurable assets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Lazy import and constructor defaults for one public factory name."""

    module: str
    class_name: str
    defaults: Mapping[str, Any] = field(default_factory=dict)


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


def _model(module: str, class_name: str, **defaults: Any) -> ModelSpec:
    return ModelSpec(module, class_name, MappingProxyType(defaults))


MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "anny": _model("body_models.anny", "ANNY"),
        "brainco": _model("body_models.brainco", "BrainCoHand"),
        "flame": _model("body_models.flame", "FLAME"),
        "g1": _model("body_models.g1", "G1"),
        "garment-measurements": _model(
            "body_models.garment_measurements",
            "GarmentMeasurements",
        ),
        "humenv": _model("body_models.smpl_humanoid", "SmplHumanoid", variant="humenv"),
        "mano": _model("body_models.mano", "MANO", side="right"),
        "mhr": _model("body_models.mhr", "MHR"),
        "myofullbody": _model("body_models.myofullbody", "MyoFullBody"),
        "phc": _model("body_models.smpl_humanoid", "SmplHumanoid", variant="phc"),
        "skel": _model("body_models.skel", "SKEL"),
        "smpl": _model("body_models.smpl", "SMPL", gender="neutral"),
        "smpl-humanoid": _model("body_models.smpl_humanoid", "SmplHumanoid"),
        "smplh": _model("body_models.smplh", "SMPLH", gender="neutral"),
        "smplsim": _model("body_models.smpl_humanoid", "SmplHumanoid", variant="smplsim"),
        "smplx": _model("body_models.smplx", "SMPLX", gender="neutral"),
        "soma": _model("body_models.soma", "SOMA"),
    }
)


def _assets(module: str, *names: str) -> dict[str, AssetSpec]:
    return {name: AssetSpec(module) for name in names}


ASSET_SPECS: Mapping[str, AssetSpec] = MappingProxyType(
    {
        **_assets("body_models.smpl._io", "smpl-male", "smpl-female", "smpl-neutral"),
        **_assets("body_models.smplx._io", "smplx-male", "smplx-female", "smplx-neutral"),
        **_assets("body_models.smplh._io", "smplh-male", "smplh-female", "smplh-neutral"),
        **_assets(
            "body_models.smpl_humanoid._io",
            "smpl-humanoid-humenv",
            "smpl-humanoid-phc",
            "smpl-humanoid-smplsim",
        ),
        **_assets("body_models.mano._io", "mano-right", "mano-left"),
        **_assets("body_models.skel._io", "skel-male", "skel-female"),
        **_assets("body_models.anny._io", "anny"),
        **_assets("body_models.mhr._io", "mhr"),
        **_assets("body_models.flame._io", "flame"),
        **_assets("body_models.brainco._io", "brainco"),
        **_assets("body_models.g1._io", "g1"),
        **_assets("body_models.soma._io", "soma"),
        **_assets("body_models.garment_measurements._io", "garment-measurements"),
        **_assets("body_models.myofullbody._io", "myofullbody"),
    }
)


def _credentials(account: str, url: str) -> CredentialSpec:
    return CredentialSpec(account, url)


DOWNLOAD_SPECS: Mapping[str, DownloadSpec] = MappingProxyType(
    {
        "smpl": DownloadSpec(
            "body_models._download",
            "download_smpl",
            credentials=_credentials("SMPL", "https://smpl.is.tue.mpg.de/"),
        ),
        "smplh": DownloadSpec(
            "body_models._download",
            "download_smplh",
            credentials=_credentials("SMPLH", "https://mano.is.tue.mpg.de/"),
        ),
        "mano": DownloadSpec(
            "body_models._download",
            "download_mano",
            credentials=_credentials("MANO", "https://mano.is.tue.mpg.de/"),
        ),
        "smplx": DownloadSpec(
            "body_models._download",
            "download_smplx",
            credentials=_credentials("SMPLX", "https://smpl-x.is.tue.mpg.de/"),
        ),
        "smpl-humanoid": DownloadSpec("body_models.smpl_humanoid._io", "download_assets"),
        "skel": DownloadSpec(
            "body_models._download",
            "download_skel_assets",
            credentials=_credentials("SKEL", "https://skel.is.tue.mpg.de/"),
        ),
        "flame": DownloadSpec(
            "body_models._download",
            "download_flame",
            output_key="flame",
            credentials=_credentials("FLAME", "https://flame.is.tue.mpg.de/"),
        ),
        "anny": DownloadSpec("body_models.anny._io", "download_model", output_key="anny"),
        "brainco": DownloadSpec("body_models.brainco._io", "download_model", output_key="brainco"),
        "mhr": DownloadSpec("body_models.mhr._io", "download_model", output_key="mhr"),
        "g1": DownloadSpec("body_models.g1._io", "download_model", output_key="g1"),
        "soma": DownloadSpec("body_models.soma._io", "download_model", output_key="soma"),
        "garment-measurements": DownloadSpec(
            "body_models.garment_measurements._io",
            "download_model",
            output_key="garment-measurements",
        ),
        "myofullbody": DownloadSpec(
            "body_models.myofullbody._io",
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
    "ModelSpec",
]
