"""SOMA identity data."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnnyIdentityData:
    template_vertices: Any
    blendshapes: Any
    phenotype_mask: Any
    anchors: dict[str, Any]


@dataclass(frozen=True)
class LinearIdentityData:
    mean: Any
    shapedirs: Any


@dataclass(frozen=True)
class MhrIdentityData:
    model_path: Any
