"""Array runtimes for backend-independent model programs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from body_models import common
from body_models.common import skinning


class Runtime(ABC):
    """Array storage and shared operation implementations for one backend."""

    name: str

    @property
    @abstractmethod
    def xp(self) -> Any:
        """Array namespace, imported lazily so runtime objects remain serializable."""

    def convert_model_data(self, value: Any) -> Any:
        """Convert model data to backend-managed state."""
        return value

    def asarray(self, value: Any, *, like: Any, dtype: Any | None = None) -> Any:
        """Create an array with the backend, device, and default dtype of ``like``."""
        if dtype is None:
            dtype = like.dtype
        return self.xp.asarray(value, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], *, like: Any, dtype: Any | None = None) -> Any:
        """Create zeros with the backend and device of ``like``."""
        return common.zeros_as(like, shape=shape, dtype=dtype, xp=self.xp)

    def compact_linear_blend_skinning(
        self,
        vertices: Any,
        transforms: Any,
        *,
        joint_indices: Any,
        joint_weights: Any,
        vertex_indices: Any | None = None,
    ) -> Any:
        """Select optional vertices and apply compact linear blend skinning."""
        if vertex_indices is not None:
            indices = self.asarray(
                vertex_indices,
                like=joint_indices,
                dtype=joint_indices.dtype,
            )
            vertices = vertices[..., indices, :]
            joint_indices = joint_indices[indices]
            joint_weights = joint_weights[indices]
        return self._compact_linear_blend_skinning(
            vertices,
            transforms,
            joint_indices=joint_indices,
            joint_weights=joint_weights,
        )

    def _compact_linear_blend_skinning(
        self,
        vertices: Any,
        transforms: Any,
        *,
        joint_indices: Any,
        joint_weights: Any,
    ) -> Any:
        """Lower compact linear blend skinning to one backend implementation."""
        return skinning.compact_linear_blend_skinning(
            vertices,
            transforms,
            joint_indices=joint_indices,
            joint_weights=joint_weights,
            xp=self.xp,
        )

    @abstractmethod
    def expand_skinning_weights(self, joint_indices: Any, joint_weights: Any, num_joints: int) -> Any:
        """Expand compact per-vertex influences into a dense weight matrix."""


class NumpyRuntime(Runtime):
    """NumPy model runtime."""

    name = "numpy"

    @property
    def xp(self) -> Any:
        import numpy as np

        return np

    def expand_skinning_weights(self, joint_indices: Any, joint_weights: Any, num_joints: int) -> Any:
        np = self.xp
        num_vertices = joint_indices.shape[0]
        rows = np.broadcast_to(np.arange(num_vertices)[:, None], joint_indices.shape)
        valid = joint_indices >= 0
        dense = np.zeros((num_vertices, num_joints), dtype=joint_weights.dtype)
        np.add.at(dense, (rows[valid], joint_indices[valid]), joint_weights[valid])
        return dense


class TorchRuntime(Runtime):
    """Torch model runtime with optional Warp operation lowerings."""

    name = "torch"
    skinning_backends = ("torch", "warp")

    def __init__(self, skinning_backend: Literal["torch", "warp"] = "torch") -> None:
        if skinning_backend not in self.skinning_backends:
            raise ValueError(f"Invalid Torch skinning backend: {skinning_backend!r}")
        self.skinning_backend = skinning_backend

    @property
    def xp(self) -> Any:
        import torch

        return torch

    def convert_model_data(self, value: Any) -> Any:
        return common.torchify(value)

    def asarray(self, value: Any, *, like: Any, dtype: Any | None = None) -> Any:
        if dtype is None:
            dtype = like.dtype
        return self.xp.as_tensor(value, device=like.device, dtype=dtype)

    def _compact_linear_blend_skinning(
        self,
        vertices: Any,
        transforms: Any,
        *,
        joint_indices: Any,
        joint_weights: Any,
    ) -> Any:
        if self.skinning_backend == "torch":
            return super()._compact_linear_blend_skinning(
                vertices,
                transforms,
                joint_indices=joint_indices,
                joint_weights=joint_weights,
            )

        try:
            from body_models.common import warp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install body-models[warp] to use skinning_backend='warp'.") from exc
        return warp.compact_linear_blend_skinning(
            vertices,
            transforms,
            joint_indices=joint_indices,
            joint_weights=joint_weights,
        )

    def expand_skinning_weights(self, joint_indices: Any, joint_weights: Any, num_joints: int) -> Any:
        torch = self.xp
        num_vertices = joint_indices.shape[0]
        dense = torch.zeros(
            (num_vertices, num_joints),
            device=joint_weights.device,
            dtype=joint_weights.dtype,
        )
        valid = joint_indices >= 0
        indices = joint_indices.clamp_min(0).long()
        weights = joint_weights * valid
        return dense.scatter_add(1, indices, weights)


class JaxRuntime(Runtime):
    """JAX model runtime."""

    name = "jax"

    @property
    def xp(self) -> Any:
        import jax.numpy as jnp

        return jnp

    def convert_model_data(self, value: Any) -> Any:
        return common.jaxify(value)

    def asarray(self, value: Any, *, like: Any, dtype: Any | None = None) -> Any:
        import jax

        if dtype is None:
            dtype = like.dtype
        array = self.xp.asarray(value, dtype=dtype)
        device = getattr(like, "device", None)
        return array if device is None else jax.device_put(array, device)

    def expand_skinning_weights(self, joint_indices: Any, joint_weights: Any, num_joints: int) -> Any:
        jnp = self.xp
        num_vertices = joint_indices.shape[0]
        rows = jnp.broadcast_to(jnp.arange(num_vertices)[:, None], joint_indices.shape)
        valid = joint_indices >= 0
        indices = jnp.maximum(joint_indices, 0)
        weights = jnp.where(valid, joint_weights, 0)
        dense = jnp.zeros((num_vertices, num_joints), dtype=joint_weights.dtype)
        return dense.at[rows, indices].add(weights)


class JaxModel:
    """Generic pytree behavior for models with ``weights`` and static ``_config``."""

    weights: Any
    _config: Any
    _runtime: JaxRuntime

    def tree_flatten(self):
        return (self.weights,), self._config

    @classmethod
    def tree_unflatten(cls, config, children):
        obj = cls.__new__(cls)
        obj._runtime = JaxRuntime()
        obj._config = config
        (obj.weights,) = children
        return obj


__all__ = ["JaxModel", "JaxRuntime", "NumpyRuntime", "Runtime", "TorchRuntime"]
