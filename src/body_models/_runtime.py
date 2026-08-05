"""Array runtimes for backend-independent model programs.

Runtime methods lower backend-independent operations at call time. Reusable
derived inputs belong to backend-materialized state instead; see
:mod:`body_models._state`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeAlias

import numpy as np
from jaxtyping import Float, Num

from body_models import _common as common
from body_models import _state as state
from body_models._common import skinning as skinning_ops

Array = Any
RuntimeName: TypeAlias = Literal["numpy", "torch", "jax"]
KernelBackend: TypeAlias = Literal["torch", "triton", "warp"]


class ArrayRuntime(ABC):
    """Shared numerical operations for one array backend."""

    name: ClassVar[RuntimeName]

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
    def _materialize(self, value: Any) -> Any:
        """Convert loaded model data into backend-managed state."""

    @abstractmethod
    def stop_gradient(self, value: Num[Array, "..."]) -> Num[Array, "..."]:
        """Return ``value`` without gradient propagation."""

    @abstractmethod
    def to_numpy(self, value: Num[Array, "..."]) -> Num[np.ndarray, "..."]:
        """Convert an array to NumPy host memory."""

    def _skin_vertices(
        self,
        vertices: Float[Array, "*batch V 3"],
        transforms: Float[Array, "*batch J 4 4"],
        *,
        skinning: skinning_ops.CompactSkinningState,
        vertex_indices: Sequence[int] | None = None,
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

    def _compose_kinematic_tree(
        self,
        local_transforms: Float[Array, "*batch J 4 4"],
        tree: common.KinematicTree,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compose local transforms along a materialized kinematic tree."""
        return common.compose_kinematic_fronts(local_transforms, tree.fronts, xp=self.xp)


@dataclass(frozen=True)
class NumpyRuntime(ArrayRuntime):
    """NumPy model runtime."""

    name = "numpy"

    @property
    def xp(self) -> Any:
        return np

    def _materialize(self, value: Any) -> Any:
        return state.numpy_state(value)

    def stop_gradient(self, value: Num[Array, "..."]) -> Num[Array, "..."]:
        return value

    def to_numpy(self, value: Num[Array, "..."]) -> Num[np.ndarray, "..."]:
        return np.asarray(value)


@dataclass(frozen=True, kw_only=True)
class TorchRuntime(ArrayRuntime):
    """Torch array runtime with optional compiled operation lowerings."""

    name = "torch"
    KERNEL_BACKENDS: ClassVar[tuple[KernelBackend, ...]] = ("torch", "triton", "warp")
    kernel_backend: KernelBackend = "torch"

    def __post_init__(self) -> None:
        if self.kernel_backend not in self.KERNEL_BACKENDS:
            raise ValueError(f"Invalid Torch kernel backend: {self.kernel_backend!r}")

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

    def _materialize(self, value: Any) -> Any:
        return state.torch_state(value, kernel_backend=self.kernel_backend)

    def stop_gradient(self, value: Num[Array, "..."]) -> Num[Array, "..."]:
        return value.detach()

    def to_numpy(self, value: Num[Array, "..."]) -> Num[np.ndarray, "..."]:
        return value.detach().cpu().numpy()

    def _skin_vertices(
        self,
        vertices: Float[Array, "*batch V 3"],
        transforms: Float[Array, "*batch J 4 4"],
        *,
        skinning: skinning_ops.CompactSkinningState,
        vertex_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch selected 3"]:
        if self.kernel_backend != "triton" or vertex_indices is None:
            return super()._skin_vertices(
                vertices,
                transforms,
                skinning=skinning,
                vertex_indices=vertex_indices,
            )

        # A subset would need a new joint-major backward plan inside the compiled call.
        output = self._compact_linear_blend_skinning(vertices, transforms, skinning=skinning)
        indices = self.asarray(vertex_indices, like=skinning.joint_indices, dtype=skinning.joint_indices.dtype)
        return output[..., indices, :]

    def _compact_linear_blend_skinning(
        self,
        vertices: Float[Array, "*batch V 3"],
        transforms: Float[Array, "*batch J 4 4"],
        *,
        skinning: skinning_ops.CompactSkinningState,
    ) -> Float[Array, "*batch V 3"]:
        backend = self.kernel_backend
        if backend == "torch":
            return super()._compact_linear_blend_skinning(
                vertices,
                transforms,
                skinning=skinning,
            )

        if backend == "triton":
            from body_models._common import triton_skinning

            if not isinstance(skinning, triton_skinning.TritonSkinningState):
                raise TypeError("Triton skinning state must be materialized before use")
            return triton_skinning.compact_linear_blend_skinning(
                vertices,
                transforms,
                skinning=skinning,
            )

        try:
            from body_models._common import warp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install body-models[warp] to use kernel_backend='warp'.") from exc
        return warp.compact_linear_blend_skinning(
            vertices,
            transforms,
            skinning=skinning,
        )


@dataclass(frozen=True)
class JaxRuntime(ArrayRuntime):
    """JAX model runtime."""

    name = "jax"

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

    def _materialize(self, value: Any) -> Any:
        return state.jax_state(value)

    def stop_gradient(self, value: Num[Array, "..."]) -> Num[Array, "..."]:
        import jax

        return jax.lax.stop_gradient(value)

    def to_numpy(self, value: Num[Array, "..."]) -> Num[np.ndarray, "..."]:
        import jax

        return np.asarray(jax.device_get(value))


RuntimeLike: TypeAlias = RuntimeName | ArrayRuntime


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
    "JaxRuntime",
    "KernelBackend",
    "NumpyRuntime",
    "RuntimeLike",
    "RuntimeName",
    "TorchRuntime",
    "resolve_runtime",
]
