"""NumPy identity sources for SOMA."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from jaxtyping import Float

from ...anny.numpy import ANNY
from ...mhr.numpy import MHR
from ...smpl.numpy import SMPL
from ...smplx.numpy import SMPLX
from ..backends import core
from ..io import SomaIdentityTransfer, get_identity_model_path
from . import anny_identity_shape, identity_transfer, linear_identity_shape, mhr_identity_shape


class IdentitySource:
    def __init__(self, transfer_data: SomaIdentityTransfer) -> None:
        self.transfer = identity_transfer(transfer_data)

    def source_shape(
        self,
        identity: Float[np.ndarray, "B I"],
        scale_params: Float[np.ndarray, "B K"] | None,
    ) -> Float[np.ndarray, "B V 3"]:
        raise NotImplementedError


class MhrIdentitySource(IdentitySource):
    def __init__(self, transfer_data: SomaIdentityTransfer) -> None:
        super().__init__(transfer_data)
        self.model = MHR(model_path=get_identity_model_path("mhr"), simplify=1.0)

    def source_shape(
        self,
        identity: Float[np.ndarray, "B I"],
        scale_params: Float[np.ndarray, "B K"] | None,
    ) -> Float[np.ndarray, "B V 3"]:
        return mhr_identity_shape(self.model, identity, scale_params, num_scale_params=68, xp=np)


class AnnyIdentitySource(IdentitySource):
    def __init__(self, transfer_data: SomaIdentityTransfer) -> None:
        super().__init__(transfer_data)
        self.model = ANNY(model_path=get_identity_model_path("anny"), all_phenotypes=False, simplify=1.0)
        rotation, translation = core.fit_rigid_transform(
            self.model.weights.template_vertices,
            transfer_data.source_vertices,
            xp=np,
        )
        self.transfer = replace(
            self.transfer,
            internal_to_source_rotation=rotation.astype(np.float32, copy=False),
            internal_to_source_translation=translation.astype(np.float32, copy=False),
            source_to_soma_rotation=np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
                dtype=np.float32,
            ),
        )

    def source_shape(
        self,
        identity: Float[np.ndarray, "B I"],
        scale_params: Float[np.ndarray, "B K"] | None,
    ) -> Float[np.ndarray, "B V 3"]:
        del scale_params
        return anny_identity_shape(
            template_vertices=self.model.weights.template_vertices,
            blendshapes=self.model.weights.blendshapes,
            phenotype_mask=self.model.weights.phenotype_mask,
            anchors=self.model.weights.anchors,
            shape=identity,
            xp=np,
        )


class LinearIdentitySource(IdentitySource):
    def __init__(self, model_type: str, transfer_data: SomaIdentityTransfer) -> None:
        super().__init__(transfer_data)
        model_cls = {"smpl": SMPL, "smplx": SMPLX}[model_type]
        self.model = model_cls(model_path=get_identity_model_path(model_type), simplify=1.0)

    def source_shape(
        self,
        identity: Float[np.ndarray, "B I"],
        scale_params: Float[np.ndarray, "B K"] | None,
    ) -> Float[np.ndarray, "B V 3"]:
        del scale_params
        return linear_identity_shape(
            mean=self.model.rest_vertices,
            shapedirs=self.model.shapedirs,
            identity=identity,
            xp=np,
        )


def create_identity_source(model_type: str, transfer_data: SomaIdentityTransfer) -> IdentitySource:
    if model_type == "mhr":
        return MhrIdentitySource(transfer_data)
    if model_type == "anny":
        return AnnyIdentitySource(transfer_data)
    return LinearIdentitySource(model_type, transfer_data)
