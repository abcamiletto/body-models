"""JAX identity sources for SOMA."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Float

from body_models.state import jax_state

from .. import core
from ..io import SomaIdentityTransfer, get_identity_model_path
from . import anny_identity_shape, identity_transfer, linear_identity_shape, mhr_identity_shape


class IdentitySource:
    def __init__(self, transfer_data: SomaIdentityTransfer) -> None:
        self.transfer = jax_state(identity_transfer(transfer_data))

    def source_shape(
        self,
        identity: Float[jax.Array, "B I"],
        scale_params: Float[jax.Array, "B K"] | None,
    ) -> Float[jax.Array, "B V 3"]:
        raise NotImplementedError

    def tree_flatten(self):
        names = tuple(self.__dict__)
        return tuple(getattr(self, name) for name in names), names

    @classmethod
    def tree_unflatten(cls, names, children):
        obj = cls.__new__(cls)
        for name, value in zip(names, children, strict=True):
            setattr(obj, name, value)
        return obj


@jax.tree_util.register_pytree_node_class
class MhrIdentitySource(IdentitySource):
    def __init__(self, transfer_data: SomaIdentityTransfer, model_factory: Callable[..., Any]) -> None:
        super().__init__(transfer_data)
        self.model = model_factory("mhr", model_path=get_identity_model_path("mhr"), simplify=1.0)

    def source_shape(
        self,
        identity: Float[jax.Array, "B I"],
        scale_params: Float[jax.Array, "B K"] | None,
    ) -> Float[jax.Array, "B V 3"]:
        return mhr_identity_shape(self.model, identity, scale_params, num_scale_params=68, xp=jnp)


@jax.tree_util.register_pytree_node_class
class AnnyIdentitySource(IdentitySource):
    def __init__(self, transfer_data: SomaIdentityTransfer, model_factory: Callable[..., Any]) -> None:
        super().__init__(transfer_data)
        self.model = model_factory(
            "anny",
            model_path=get_identity_model_path("anny"),
            all_phenotypes=False,
            simplify=1.0,
        )
        rotation, translation = core.fit_rigid_transform(
            self.model.weights.template_vertices[...],
            jnp.asarray(transfer_data.source_vertices),
            xp=jnp,
        )
        self.transfer = replace(
            self.transfer,
            internal_to_source_rotation=rotation,
            internal_to_source_translation=translation,
            source_to_soma_rotation=jnp.asarray(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            ),
        )

    def source_shape(
        self,
        identity: Float[jax.Array, "B I"],
        scale_params: Float[jax.Array, "B K"] | None,
    ) -> Float[jax.Array, "B V 3"]:
        del scale_params
        return anny_identity_shape(
            template_vertices=self.model.weights.template_vertices[...],
            blendshapes=self.model.weights.blendshapes[...],
            phenotype_mask=self.model.weights.phenotype_mask[...],
            anchors=self.model.weights.anchors,
            shape=identity,
            xp=jnp,
        )


@jax.tree_util.register_pytree_node_class
class LinearIdentitySource(IdentitySource):
    def __init__(
        self,
        model_type: str,
        transfer_data: SomaIdentityTransfer,
        model_factory: Callable[..., Any],
    ) -> None:
        super().__init__(transfer_data)
        self.model = model_factory(
            model_type,
            model_path=get_identity_model_path(model_type),
            gender=None,
            simplify=1.0,
        )

    def source_shape(
        self,
        identity: Float[jax.Array, "B I"],
        scale_params: Float[jax.Array, "B K"] | None,
    ) -> Float[jax.Array, "B V 3"]:
        del scale_params
        return linear_identity_shape(
            mean=self.model.rest_vertices,
            shapedirs=self.model.shapedirs,
            identity=identity,
            xp=jnp,
        )


def create_identity_source(
    model_type: str,
    transfer_data: SomaIdentityTransfer,
    *,
    model_factory: Callable[..., Any],
) -> IdentitySource:
    if model_type == "mhr":
        return MhrIdentitySource(transfer_data, model_factory)
    if model_type == "anny":
        return AnnyIdentitySource(transfer_data, model_factory)
    return LinearIdentitySource(model_type, transfer_data, model_factory)
