"""Anny identity setup for SOMA."""

from dataclasses import dataclass, replace

import numpy as np
from jaxtyping import Float
from typing import Any

from ...anny import core as anny_core
from ...anny.numpy import ANNY
from .geometry import fit_rigid_transform
from ..io import SomaIdentityTransfer, get_identity_model_path


@dataclass(frozen=True)
class AnnyIdentity:
    template_vertices: Float[np.ndarray, "V 3"]
    blendshapes: Float[np.ndarray, "S V 3"]
    phenotype_mask: Float[np.ndarray, "S P"]
    anchors: dict[str, Float[np.ndarray, "A"]]


def prepare(transfer: SomaIdentityTransfer) -> tuple[AnnyIdentity, SomaIdentityTransfer]:
    model = ANNY(model_path=get_identity_model_path("anny"), all_phenotypes=False, simplify=1.0)
    identity_model = AnnyIdentity(
        template_vertices=model.template_vertices,
        blendshapes=model.blendshapes,
        phenotype_mask=model.phenotype_mask,
        anchors=model._anchors,
    )
    source_vertices = np.asarray(transfer.source_vertices, dtype=np.float32)
    rotation, translation = fit_rigid_transform(
        identity_model.template_vertices,
        source_vertices,
    )
    transfer = replace(
        transfer,
        internal_to_source_rotation=rotation.astype(np.float32, copy=False),
        internal_to_source_translation=translation.astype(np.float32, copy=False),
        source_to_soma_rotation=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        ),
    )
    return identity_model, transfer


def shape(
    identity_model: AnnyIdentity,
    identity: Float[Any, "B 6"],
    *,
    xp: Any,
) -> Float[Any, "B V 3"]:
    return anny_core.identity_shape(
        template_vertices=identity_model.template_vertices,
        blendshapes=identity_model.blendshapes,
        phenotype_mask=identity_model.phenotype_mask,
        anchors=identity_model.anchors,
        identity=identity,
        xp=xp,
    )
