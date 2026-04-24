"""JAX backend for the GarmentMeasurements PCA body model."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Float, Int
from nanomanifold import SO3

from ..rotations import VALID_ROTATION_TYPES
from . import core
from .io import load_model_data

__all__ = ["GarmentMeasurements"]


class GarmentMeasurements(nnx.Module):
    """GarmentMeasurements PCA body model with JAX/Flax NNX backend."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: core.RotationType = "axis_angle",
    ) -> None:
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")

        data = load_model_data(model_path=model_path, dtype=np.float32)
        self.mean_vertices = nnx.Variable(jnp.asarray(data["mean_vertices"]))
        self.components = nnx.Variable(jnp.asarray(data["components"]))
        self.eigenvalues = nnx.Variable(jnp.asarray(data["eigenvalues"]))
        self._faces = nnx.Variable(jnp.asarray(data["faces"]))
        self.rotation_type = rotation_type

    @property
    def faces(self) -> Int[jax.Array, "F _"]:
        return self._faces[...]

    @property
    def num_vertices(self) -> int:
        return self.mean_vertices[...].shape[0]

    @property
    def num_shape_components(self) -> int:
        return self.eigenvalues[...].shape[0]

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.mean_vertices[...]

    def forward_vertices(
        self,
        shape: Float[jax.Array, "B C"],
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        vertex_indices: list[int] | None = None,
    ) -> Float[jax.Array, "B V 3"]:
        return core.forward_vertices(
            mean_vertices=self.mean_vertices[...],
            components=self.components[...],
            eigenvalues=self.eigenvalues[...],
            shape=shape,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jax.Array]:
        zeros = jnp.zeros((batch_size,), dtype=dtype)
        return {
            "shape": jnp.zeros((batch_size, self.num_shape_components), dtype=dtype),
            "global_rotation": SO3.identity_as(
                zeros,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }
