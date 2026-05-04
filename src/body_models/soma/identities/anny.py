"""Anny identity setup for SOMA."""

from dataclasses import replace

import numpy as np

from ...anny.numpy import ANNY
from .base import AnnyIdentityData
from .geometry import fit_rigid_transform
from ..io import SomaIdentityTransfer, get_identity_model_path


def prepare(transfer: SomaIdentityTransfer) -> tuple[AnnyIdentityData, SomaIdentityTransfer]:
    model = ANNY(model_path=get_identity_model_path("anny"), all_phenotypes=False, simplify=1.0)
    identity_model = AnnyIdentityData(
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
