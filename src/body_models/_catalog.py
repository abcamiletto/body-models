"""Authoritative catalog of public model families."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType, ModuleType
from typing import Any


@dataclass(frozen=True)
class LicensedArchive:
    """Credentialed MPI download for a licensed model family."""

    account: str
    """Credential label and prefix of the ``<ACCOUNT>_USERNAME``/``<ACCOUNT>_PASSWORD`` variables."""
    site: str
    """Registration page shown when asking for credentials."""
    url: str
    files: Mapping[str, str]
    """Asset config key to the file name searched for in the extracted archive."""


@dataclass(frozen=True)
class ModelFamily:
    """
    Everything the package knows about one model family.

    Assets are validated by ``<module>._io.validate_path``. Families without a
    licensed archive fetch public assets with ``<module>._io.download_model``.
    """

    name: str
    package: str
    class_name: str
    defaults: Mapping[str, Any]
    licensed: LicensedArchive | None = None

    @property
    def module(self) -> str:
        return f"body_models.{self.package}"

    @property
    def asset_keys(self) -> tuple[str, ...]:
        """Config keys for this family's assets; the one loaded by ``defaults`` comes first."""
        return (self.name,) if self.licensed is None else tuple(self.licensed.files)

    def io(self) -> ModuleType:
        return import_module(f"{self.module}._io")


def _family(
    name: str,
    class_name: str,
    *,
    package: str | None = None,
    licensed: LicensedArchive | None = None,
    **defaults: Any,
) -> ModelFamily:
    return ModelFamily(name, package or name, class_name, MappingProxyType(defaults), licensed)


def _mpi(account: str, site: str, url: str, files: dict[str, str]) -> LicensedArchive:
    return LicensedArchive(account, site, url, MappingProxyType(files))


_FAMILIES = (
    _family("anny", "ANNY"),
    _family(
        "flame",
        "FLAME",
        licensed=_mpi(
            "FLAME",
            "https://flame.is.tue.mpg.de/",
            "https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2023.zip&resume=1",
            {"flame": "flame2023.pkl"},
        ),
    ),
    _family("garment-measurements", "GarmentMeasurements", package="garment_measurements"),
    _family("gnm", "GNM"),
    _family(
        "mano",
        "MANO",
        side="right",
        licensed=_mpi(
            "MANO",
            "https://mano.is.tue.mpg.de/",
            "https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=mano_v1_2.zip",
            {"mano-right": "MANO_RIGHT.pkl", "mano-left": "MANO_LEFT.pkl"},
        ),
    ),
    _family("mhr", "MHR"),
    _family(
        "skel",
        "SKEL",
        licensed=_mpi(
            "SKEL",
            "https://skel.is.tue.mpg.de/",
            "https://download.is.tue.mpg.de/download.php?domain=skel&sfile=skel_models_v1.1.zip&resume=1",
            {"skel-male": "skel_male.pkl", "skel-female": "skel_female.pkl"},
        ),
    ),
    _family(
        "smpl",
        "SMPL",
        gender="neutral",
        licensed=_mpi(
            "SMPL",
            "https://smpl.is.tue.mpg.de/",
            "https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip",
            {
                "smpl-neutral": "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
                "smpl-female": "basicmodel_f_lbs_10_207_0_v1.1.0.pkl",
                "smpl-male": "basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
            },
        ),
    ),
    _family(
        "smplh",
        "SMPLH",
        gender="neutral",
        licensed=_mpi(
            "SMPLH",
            "https://mano.is.tue.mpg.de/",
            "https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=smplh.tar.xz",
            {
                "smplh-neutral": "neutral/model.npz",
                "smplh-female": "female/model.npz",
                "smplh-male": "male/model.npz",
            },
        ),
    ),
    _family(
        "smplx",
        "SMPLX",
        gender="neutral",
        licensed=_mpi(
            "SMPLX",
            "https://smpl-x.is.tue.mpg.de/",
            "https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip",
            {
                "smplx-neutral": "SMPLX_NEUTRAL.npz",
                "smplx-female": "SMPLX_FEMALE.npz",
                "smplx-male": "SMPLX_MALE.npz",
            },
        ),
    ),
    _family("soma", "SOMA"),
)

FAMILIES: Mapping[str, ModelFamily] = MappingProxyType({family.name: family for family in _FAMILIES})
"""Model families by public catalog name."""

ASSET_FAMILIES: Mapping[str, ModelFamily] = MappingProxyType(
    {key: family for family in _FAMILIES for key in family.asset_keys}
)
"""Model families by asset config key."""


__all__ = ["ASSET_FAMILIES", "FAMILIES", "LicensedArchive", "ModelFamily"]
