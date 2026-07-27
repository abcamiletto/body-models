"""Array runtimes for backend-independent model programs.

Runtime methods lower backend-independent operations at call time. Reusable
derived inputs belong to backend-materialized state instead; see
:mod:`body_models._state`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeAlias

import numpy as np
from jaxtyping import Float, Int, Num

from body_models import _common as common
from body_models import _state as state
from body_models._common import skinning as skinning_ops

Array = Any
Backend = Literal["numpy", "torch", "jax"]


class ArrayRuntime(ABC):
    """Shared numerical operations for one array backend."""

    backend: ClassVar[Backend]

    @property
    @abstractmethod
    def xp(self) -> Any:
        """Array namespace for this runtime."""

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

    @abstractmethod
    def materialize(self, value: Any) -> Any:
        """Convert loaded model data into backend-managed state."""

    @abstractmethod
    def stop_gradient(self, value: Num[Array, "..."]) -> Num[Array, "..."]:
        """Return ``value`` without gradient propagation."""

    @abstractmethod
    def to_numpy(self, value: Num[Array, "..."]) -> Num[np.ndarray, "..."]:
        """Convert an array to NumPy host memory."""

    def compact_linear_blend_skinning(
        self,
        vertices: Float[Array, "*batch V 3"],
        transforms: Float[Array, "*batch J 4 4"],
        *,
        skinning: skinning_ops.CompactSkinningState,
        vertex_indices: Int[Array, "S"] | None = None,
    ) -> Float[Array, "*batch selected 3"]:
        """Select optional vertices and apply compact linear blend skinning."""
        if vertex_indices is not None:
            indices = self.asarray(
                vertex_indices,
                like=skinning.joint_indices,
                dtype=skinning.joint_indices.dtype,
            )
            vertices = vertices[..., indices, :]
            skinning = skinning_ops.CompactSkinning(
                joint_indices=skinning.joint_indices[indices],
                joint_weights=skinning.joint_weights[indices],
            )
        return self._compact_linear_blend_skinning(
            vertices,
            transforms,
            skinning=skinning,
        )

    def _compact_linear_blend_skinning(
        self,
        vertices: Float[Array, "*batch V 3"],
        transforms: Float[Array, "*batch J 4 4"],
        *,
        skinning: skinning_ops.CompactSkinningState,
    ) -> Float[Array, "*batch V 3"]:
        """Lower compact linear blend skinning to one backend implementation."""
        return skinning_ops.compact_linear_blend_skinning(
            vertices,
            transforms,
            joint_indices=skinning.joint_indices,
            joint_weights=skinning.joint_weights,
            xp=self.xp,
        )


@dataclass(frozen=True)
class NumpyRuntime(ArrayRuntime):
    """NumPy model runtime."""

    backend = "numpy"

    @property
    def xp(self) -> Any:
        return np

    def materialize(self, value: Any) -> Any:
        return state.numpy_state(value)

    def stop_gradient(self, value: Num[Array, "..."]) -> Num[Array, "..."]:
        return value

    def to_numpy(self, value: Num[Array, "..."]) -> Num[np.ndarray, "..."]:
        return np.asarray(value)


@dataclass(frozen=True)
class TorchRuntime(ArrayRuntime):
    """Torch model runtime with optional Warp operation lowerings."""

    backend = "torch"
    SKINNING_BACKENDS = ("torch", "warp")
    skinning_backend: Literal["torch", "warp"] = "torch"

    def __post_init__(self) -> None:
        if self.skinning_backend not in self.SKINNING_BACKENDS:
            raise ValueError(f"Invalid Torch skinning backend: {self.skinning_backend!r}")

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

    def materialize(self, value: Any) -> Any:
        return state.torch_state(value, skinning_backend=self.skinning_backend)

    def stop_gradient(self, value: Num[Array, "..."]) -> Num[Array, "..."]:
        return value.detach()

    def to_numpy(self, value: Num[Array, "..."]) -> Num[np.ndarray, "..."]:
        return value.detach().cpu().numpy()

    def _compact_linear_blend_skinning(
        self,
        vertices: Float[Array, "*batch V 3"],
        transforms: Float[Array, "*batch J 4 4"],
        *,
        skinning: skinning_ops.CompactSkinningState,
    ) -> Float[Array, "*batch V 3"]:
        if self.skinning_backend == "torch":
            return super()._compact_linear_blend_skinning(
                vertices,
                transforms,
                skinning=skinning,
            )

        try:
            from body_models._common import warp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install body-models[warp] to use skinning_backend='warp'.") from exc
        return warp.compact_linear_blend_skinning(
            vertices,
            transforms,
            skinning=skinning,
        )


@dataclass(frozen=True)
class JaxRuntime(ArrayRuntime):
    """JAX model runtime."""

    backend = "jax"

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

    def materialize(self, value: Any) -> Any:
        return state.jax_state(value)

    def stop_gradient(self, value: Num[Array, "..."]) -> Num[Array, "..."]:
        import jax

        return jax.lax.stop_gradient(value)

    def to_numpy(self, value: Num[Array, "..."]) -> Num[np.ndarray, "..."]:
        import jax

        return np.asarray(jax.device_get(value))


RuntimeLike: TypeAlias = Backend | ArrayRuntime


def resolve_runtime(runtime: RuntimeLike) -> ArrayRuntime:
    """Resolve a runtime name while preserving explicitly configured runtimes."""
    if isinstance(runtime, ArrayRuntime):
        return runtime
    if runtime == "numpy":
        return NumpyRuntime()
    if runtime == "torch":
        return TorchRuntime()
    if runtime == "jax":
        return JaxRuntime()
    raise ValueError(f"Unknown runtime {runtime!r}. Expected numpy, torch, or jax.")


__all__ = [
    "ArrayRuntime",
    "Backend",
    "JaxRuntime",
    "NumpyRuntime",
    "RuntimeLike",
    "TorchRuntime",
    "resolve_runtime",
]
