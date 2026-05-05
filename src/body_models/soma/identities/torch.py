"""PyTorch identity sources for SOMA."""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from ...anny.torch import ANNY
from ...mhr.torch import MHR
from ...smpl.torch import SMPL
from ...smplx.torch import SMPLX
from .. import core
from ..io import SomaIdentityTransfer, get_identity_model_path
from . import IdentityTransfer, anny_identity_shape, linear_identity_shape, mhr_identity_shape


class IdentitySource(nn.Module):
    def __init__(self, transfer_data: SomaIdentityTransfer) -> None:
        super().__init__()
        self.source_scale = transfer_data.source_scale
        self.output_scale = transfer_data.output_scale
        self.register_buffer("source_tetrahedra", torch.as_tensor(transfer_data.source_tetrahedra, dtype=torch.int64))
        self.register_buffer("face_ids", torch.as_tensor(transfer_data.face_ids, dtype=torch.int64))
        self.register_buffer("bary_coords", torch.as_tensor(transfer_data.bary_coords))
        self.register_buffer("unknown_ids", torch.as_tensor(transfer_data.unknown_ids, dtype=torch.int64))
        self.register_buffer("anchor_ids", torch.as_tensor(transfer_data.anchor_ids, dtype=torch.int64))
        self.register_buffer("solve_matrix", torch.as_tensor(transfer_data.solve_matrix))
        self.register_buffer("anchor_matrix", torch.as_tensor(transfer_data.anchor_matrix))
        self.register_buffer("rhs_base", torch.as_tensor(transfer_data.rhs_base))
        self.register_buffer("internal_to_source_rotation", torch.as_tensor(transfer_data.internal_to_source_rotation))
        self.register_buffer(
            "internal_to_source_translation", torch.as_tensor(transfer_data.internal_to_source_translation)
        )
        self.register_buffer("source_to_soma_rotation", torch.as_tensor(transfer_data.source_to_soma_rotation))

    @property
    def transfer(self) -> IdentityTransfer:
        return IdentityTransfer(
            source_tetrahedra=self.source_tetrahedra,
            face_ids=self.face_ids,
            bary_coords=self.bary_coords,
            unknown_ids=self.unknown_ids,
            anchor_ids=self.anchor_ids,
            solve_matrix=self.solve_matrix,
            anchor_matrix=self.anchor_matrix,
            rhs_base=self.rhs_base,
            internal_to_source_rotation=self.internal_to_source_rotation,
            internal_to_source_translation=self.internal_to_source_translation,
            source_to_soma_rotation=self.source_to_soma_rotation,
            source_scale=self.source_scale,
            output_scale=self.output_scale,
        )

    def source_shape(
        self,
        identity: Float[Tensor, "B I"],
        scale_params: Float[Tensor, "B K"] | None,
    ) -> Float[Tensor, "B V 3"]:
        raise NotImplementedError


class MhrIdentitySource(IdentitySource):
    def __init__(self, transfer_data: SomaIdentityTransfer) -> None:
        super().__init__(transfer_data)
        self.model = MHR(model_path=get_identity_model_path("mhr"), simplify=1.0)

    def source_shape(
        self,
        identity: Float[Tensor, "B I"],
        scale_params: Float[Tensor, "B K"] | None,
    ) -> Float[Tensor, "B V 3"]:
        return mhr_identity_shape(self.model, identity, scale_params, num_scale_params=68, xp=torch)


class AnnyIdentitySource(IdentitySource):
    def __init__(self, transfer_data: SomaIdentityTransfer) -> None:
        super().__init__(transfer_data)
        self.model = ANNY(model_path=get_identity_model_path("anny"), all_phenotypes=False, simplify=1.0)
        source_vertices = torch.as_tensor(transfer_data.source_vertices, dtype=self.model.template_vertices.dtype)
        rotation, translation = core.fit_rigid_transform(self.model.template_vertices, source_vertices, xp=torch)
        self.internal_to_source_rotation = rotation
        self.internal_to_source_translation = translation
        self.source_to_soma_rotation = torch.as_tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=self.model.template_vertices.dtype,
        )

    def source_shape(
        self,
        identity: Float[Tensor, "B I"],
        scale_params: Float[Tensor, "B K"] | None,
    ) -> Float[Tensor, "B V 3"]:
        del scale_params
        return anny_identity_shape(
            template_vertices=self.model.template_vertices,
            blendshapes=self.model.blendshapes,
            phenotype_mask=self.model.phenotype_mask,
            anchors=self.model._get_anchors_dict(),
            identity=identity,
            xp=torch,
        )


class LinearIdentitySource(IdentitySource):
    def __init__(self, model_type: str, transfer_data: SomaIdentityTransfer) -> None:
        super().__init__(transfer_data)
        model_cls = {"smpl": SMPL, "smplx": SMPLX}[model_type]
        self.model = model_cls(model_path=get_identity_model_path(model_type), simplify=1.0)

    def source_shape(
        self,
        identity: Float[Tensor, "B I"],
        scale_params: Float[Tensor, "B K"] | None,
    ) -> Float[Tensor, "B V 3"]:
        del scale_params
        return linear_identity_shape(
            mean=self.model.v_template_full,
            shapedirs=self.model.shapedirs_full,
            identity=identity,
            xp=torch,
        )


def create_identity_source(model_type: str, transfer_data: SomaIdentityTransfer) -> IdentitySource:
    if model_type == "mhr":
        return MhrIdentitySource(transfer_data)
    if model_type == "anny":
        return AnnyIdentitySource(transfer_data)
    return LinearIdentitySource(model_type, transfer_data)
