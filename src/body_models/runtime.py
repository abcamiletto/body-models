"""Array runtimes for backend-independent model programs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from jaxtyping import Float, Int, Num

from body_models import common
from body_models.common import skinning

Array = Any


class ArrayRuntime(ABC):
    """Shared numerical operations for one array backend."""

    @property
    @abstractmethod
    def xp(self) -> Any:
        """Array namespace, imported lazily so runtime objects remain serializable."""

    def asarray(
        self,
        value: Any,
        *,
        like: Num[Array, "..."],
        dtype: Any | None = None,
    ) -> Num[Array, "..."]:
        """Create an array with the backend, device, and default dtype of ``like``."""
        if dtype is None:
            dtype = like.dtype
        return self.xp.asarray(value, dtype=dtype)

    def zeros(
        self,
        shape: tuple[int, ...],
        *,
        like: Float[Array, "..."],
        dtype: Any | None = None,
    ) -> Float[Array, "..."]:
        """Create zeros with the backend and device of ``like``."""
        return common.zeros_as(like, shape=shape, dtype=dtype, xp=self.xp)

    def compact_linear_blend_skinning(
        self,
        vertices: Float[Array, "*batch V 3"],
        transforms: Float[Array, "*batch J 4 4"],
        *,
        joint_indices: Int[Array, "V K"],
        joint_weights: Float[Array, "V K"],
        vertex_indices: Int[Array, "S"] | None = None,
    ) -> Float[Array, "*batch selected 3"]:
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
        vertices: Float[Array, "*batch V 3"],
        transforms: Float[Array, "*batch J 4 4"],
        *,
        joint_indices: Int[Array, "V K"],
        joint_weights: Float[Array, "V K"],
    ) -> Float[Array, "*batch V 3"]:
        """Lower compact linear blend skinning to one backend implementation."""
        return skinning.compact_linear_blend_skinning(
            vertices,
            transforms,
            joint_indices=joint_indices,
            joint_weights=joint_weights,
            xp=self.xp,
        )


class NumpyRuntime(ArrayRuntime):
    """NumPy model runtime."""

    @property
    def xp(self) -> Any:
        import numpy as np

        return np


class TorchRuntime(ArrayRuntime):
    """Torch model runtime with optional Warp operation lowerings."""

    SKINNING_BACKENDS = ("torch", "warp")

    def __init__(self, skinning_backend: Literal["torch", "warp"] = "torch") -> None:
        if skinning_backend not in self.SKINNING_BACKENDS:
            raise ValueError(f"Invalid Torch skinning backend: {skinning_backend!r}")
        self.skinning_backend = skinning_backend

    @property
    def xp(self) -> Any:
        import torch

        return torch

    def asarray(
        self,
        value: Any,
        *,
        like: Num[Array, "..."],
        dtype: Any | None = None,
    ) -> Num[Array, "..."]:
        if dtype is None:
            dtype = like.dtype
        return self.xp.as_tensor(value, device=like.device, dtype=dtype)

    def _compact_linear_blend_skinning(
        self,
        vertices: Float[Array, "*batch V 3"],
        transforms: Float[Array, "*batch J 4 4"],
        *,
        joint_indices: Int[Array, "V K"],
        joint_weights: Float[Array, "V K"],
    ) -> Float[Array, "*batch V 3"]:
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


class JaxRuntime(ArrayRuntime):
    """JAX model runtime."""

    @property
    def xp(self) -> Any:
        import jax.numpy as jnp

        return jnp

    def asarray(
        self,
        value: Any,
        *,
        like: Num[Array, "..."],
        dtype: Any | None = None,
    ) -> Num[Array, "..."]:
        import jax

        if dtype is None:
            dtype = like.dtype
        array = self.xp.asarray(value, dtype=dtype)
        device = getattr(like, "device", None)
        return array if device is None else jax.device_put(array, device)


class JaxModel:
    """Pytree contract for models with array ``weights`` and static ``_config``."""

    _jax_children: ClassVar[tuple[str, ...]] = ("weights",)
    weights: Any
    _config: Any
    _runtime: JaxRuntime

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in self._jax_children), self._config

    def _rebuild_jax_state(self) -> None:
        """Restore state derived from pytree children after reconstruction."""

    @classmethod
    def tree_unflatten(cls, config, children):
        obj = cls.__new__(cls)
        obj._runtime = JaxRuntime()
        obj._config = config
        for name, value in zip(cls._jax_children, children, strict=True):
            setattr(obj, name, value)
        obj._rebuild_jax_state()
        return obj


__all__ = ["ArrayRuntime", "JaxModel", "JaxRuntime", "NumpyRuntime", "TorchRuntime"]
