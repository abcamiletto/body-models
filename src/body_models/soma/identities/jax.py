"""JAX identity sources for SOMA."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Float

from ...anny.jax import ANNY
from ...mhr.jax import MHR
from ...smpl.jax import SMPL
from ...smplx.jax import SMPLX
from ..backend import core
from ..io import SomaIdentityTransfer, get_identity_model_path
from . import IdentityTransfer, anny_identity_shape, linear_identity_shape, mhr_identity_shape


class IdentitySource(nnx.Module):
    def __init__(self, transfer_data: SomaIdentityTransfer) -> None:
        self.source_scale = transfer_data.source_scale
        self.output_scale = transfer_data.output_scale
        self.source_tetrahedra = nnx.Variable(jnp.asarray(transfer_data.source_tetrahedra))
        self.face_ids = nnx.Variable(jnp.asarray(transfer_data.face_ids))
        self.bary_coords = nnx.Variable(jnp.asarray(transfer_data.bary_coords))
        self.unknown_ids = nnx.Variable(jnp.asarray(transfer_data.unknown_ids))
        self.anchor_ids = nnx.Variable(jnp.asarray(transfer_data.anchor_ids))
        self.solve_matrix = nnx.Variable(jnp.asarray(transfer_data.solve_matrix))
        self.anchor_matrix = nnx.Variable(jnp.asarray(transfer_data.anchor_matrix))
        self.rhs_base = nnx.Variable(jnp.asarray(transfer_data.rhs_base))
        self.internal_to_source_rotation = nnx.Variable(jnp.asarray(transfer_data.internal_to_source_rotation))
        self.internal_to_source_translation = nnx.Variable(jnp.asarray(transfer_data.internal_to_source_translation))
        self.source_to_soma_rotation = nnx.Variable(jnp.asarray(transfer_data.source_to_soma_rotation))

    @property
    def transfer(self) -> IdentityTransfer:
        return IdentityTransfer(
            source_tetrahedra=self.source_tetrahedra[...],
            face_ids=self.face_ids[...],
            bary_coords=self.bary_coords[...],
            unknown_ids=self.unknown_ids[...],
            anchor_ids=self.anchor_ids[...],
            solve_matrix=self.solve_matrix[...],
            anchor_matrix=self.anchor_matrix[...],
            rhs_base=self.rhs_base[...],
            internal_to_source_rotation=self.internal_to_source_rotation[...],
            internal_to_source_translation=self.internal_to_source_translation[...],
            source_to_soma_rotation=self.source_to_soma_rotation[...],
            source_scale=self.source_scale,
            output_scale=self.output_scale,
        )

    def source_shape(
        self,
        identity: Float[jax.Array, "B I"],
        scale_params: Float[jax.Array, "B K"] | None,
    ) -> Float[jax.Array, "B V 3"]:
        raise NotImplementedError


class MhrIdentitySource(IdentitySource):
    def __init__(self, transfer_data: SomaIdentityTransfer) -> None:
        super().__init__(transfer_data)
        self.model = nnx.data(MHR(model_path=get_identity_model_path("mhr"), simplify=1.0))

    def source_shape(
        self,
        identity: Float[jax.Array, "B I"],
        scale_params: Float[jax.Array, "B K"] | None,
    ) -> Float[jax.Array, "B V 3"]:
        return mhr_identity_shape(self.model, identity, scale_params, num_scale_params=68, xp=jnp)


class AnnyIdentitySource(IdentitySource):
    def __init__(self, transfer_data: SomaIdentityTransfer) -> None:
        super().__init__(transfer_data)
        self.model = nnx.data(ANNY(model_path=get_identity_model_path("anny"), all_phenotypes=False, simplify=1.0))
        rotation, translation = core.fit_rigid_transform(
            self.model.template_vertices[...],
            jnp.asarray(transfer_data.source_vertices),
            xp=jnp,
        )
        self.internal_to_source_rotation = nnx.Variable(rotation)
        self.internal_to_source_translation = nnx.Variable(translation)
        self.source_to_soma_rotation = nnx.Variable(jnp.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]))

    def source_shape(
        self,
        identity: Float[jax.Array, "B I"],
        scale_params: Float[jax.Array, "B K"] | None,
    ) -> Float[jax.Array, "B V 3"]:
        del scale_params
        return anny_identity_shape(
            template_vertices=self.model.template_vertices[...],
            blendshapes=self.model.blendshapes[...],
            phenotype_mask=self.model.phenotype_mask[...],
            anchors=self.model._get_anchors_dict(),
            identity=identity,
            xp=jnp,
        )


class LinearIdentitySource(IdentitySource):
    def __init__(self, model_type: str, transfer_data: SomaIdentityTransfer) -> None:
        super().__init__(transfer_data)
        model_cls = {"smpl": SMPL, "smplx": SMPLX}[model_type]
        self.model = nnx.data(model_cls(model_path=get_identity_model_path(model_type), simplify=1.0))

    def source_shape(
        self,
        identity: Float[jax.Array, "B I"],
        scale_params: Float[jax.Array, "B K"] | None,
    ) -> Float[jax.Array, "B V 3"]:
        del scale_params
        return linear_identity_shape(
            mean=self.model.v_template_full[...],
            shapedirs=self.model.shapedirs_full[...],
            identity=identity,
            xp=jnp,
        )


def create_identity_source(model_type: str, transfer_data: SomaIdentityTransfer) -> IdentitySource:
    if model_type == "mhr":
        return MhrIdentitySource(transfer_data)
    if model_type == "anny":
        return AnnyIdentitySource(transfer_data)
    return LinearIdentitySource(model_type, transfer_data)
