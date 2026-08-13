from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any, ClassVar, Literal

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import _pose_layout as pose_layout
from body_models import _state as state
from body_models._common import deformation, point_regression, skinning
from body_models._constants import Joint
from body_models._rotations import RotationType, rotation_ndim, rotation_shape
from body_models._runtime import ArrayRuntime

Array = Any
ParameterRole = Literal["identity", "pose", "transform"]


@dataclass(frozen=True)
class ParameterSpec:
    """Shape, role, and numeric default of one model parameter."""

    shape: tuple[int, ...]
    role: ParameterRole
    default: float = field(default=0.0, kw_only=True)
    rotation_type: RotationType | None = field(default=None, kw_only=True)

    @classmethod
    def rotation(
        cls,
        rotation_type: RotationType,
        *,
        count: int | None = None,
        role: ParameterRole = "pose",
    ) -> ParameterSpec:
        """Describe one rotation or a vector of rotations."""
        leading_shape = () if count is None else (count,)
        return cls(
            shape=(*leading_shape, *rotation_shape(rotation_type)),
            role=role,
            rotation_type=rotation_type,
        )


CorrectiveBasis = deformation.CorrectiveBasis
DenseCorrectiveBasis = deformation.DenseCorrectiveBasis
LinearIdentity = deformation.LinearIdentity
PointRegressor = point_regression.PointRegressor
SkinningSpec = deformation.SkinningSpec
SkinningIdentity = deformation.SkinningIdentity
SkinningPose = deformation.SkinningPose
SparseCorrectiveBasis = deformation.SparseCorrectiveBasis


class SkinnedModel(ABC):
    """Base class for skinned body models."""

    _COMMON_JOINTS: ClassVar[Mapping[Joint, str]] = {}
    _POSE_LAYOUT: ClassVar[pose_layout.PoseLayout | None] = None
    _state_fields: ClassVar[tuple[str, ...]] = ("_weights",)
    _config: Any
    _runtime: ArrayRuntime
    has_face: ClassVar[bool] = False
    has_hands: ClassVar[bool] = False

    @property
    def runtime(self) -> ArrayRuntime:
        """Array runtime used by this model."""
        return self._runtime

    def _attach_runtime(self, runtime: ArrayRuntime) -> None:
        self._runtime = runtime
        if runtime.name == "jax":
            _register_jax_model(type(self))

    def __setstate__(self, values: dict[str, Any]) -> None:
        self.__dict__.update(values)
        if self.runtime.name != "jax":
            return
        _register_jax_model(type(self))
        state.register_jax_state(tuple(getattr(self, name) for name in self._state_fields))

    @property
    @abstractmethod
    def faces(self) -> Int[Array, "F C"]:
        """Mesh face indices. Shape [F, 3] for triangles or [F, 4] for quads."""

    @property
    def num_joints(self) -> int:
        """Number of joints in the skeleton."""
        return len(self.parents)

    @property
    @abstractmethod
    def num_vertices(self) -> int:
        """Number of mesh vertices."""

    @property
    @abstractmethod
    def joint_names(self) -> list[str]:
        """Joint names in joint index order."""

    @property
    @abstractmethod
    def parents(self) -> list[int]:
        """Parent indices in joint_names order, with -1 for the root."""

    @property
    def common_joints(self) -> Mapping[Joint, str]:
        """Common anatomical joints mapped to this model's native joint names."""
        return self._COMMON_JOINTS

    @property
    def pose_joint_indices(self) -> Mapping[str, tuple[int, ...]]:
        """Canonical joints whose local transforms are driven by each pose parameter."""
        layout = self._pose_layout
        if layout is None:
            return {}
        pose_parameters = {name for name, spec in self.parameter_spec.items() if spec.role == "pose"}
        joint_indices = layout.joint_indices
        if set(joint_indices) != pose_parameters:
            raise ValueError("Pose layout does not match pose parameters")
        return joint_indices

    @property
    def _pose_layout(self) -> pose_layout.PoseLayout | None:
        return self._POSE_LAYOUT

    def joint_index(self, joint: Joint) -> int:
        """Resolve a common joint to this model's native joint index."""
        if not isinstance(joint, Joint):
            raise TypeError("joint_index() expects a body_models.Joint; use joint_names.index(...) for native names.")
        try:
            native_name = self.common_joints[joint]
        except KeyError as exc:
            raise KeyError(f"{self.__class__.__name__} has no common joint {joint.value!r}") from exc
        return self.joint_names.index(native_name)

    @property
    @abstractmethod
    def parameter_spec(self) -> Mapping[str, ParameterSpec]:
        """Machine-readable parameters accepted by this model."""

    @property
    @abstractmethod
    def _parameter_reference(self) -> Float[Array, "..."]:
        """Array whose backend, device, and dtype parameter defaults follow."""

    @abstractmethod
    def forward_skeleton(self, *args, **kwargs) -> Float[Array, "*batch J 4 4"]:
        """
        Compute skeleton joint transforms.

        Signatures vary by model. Outputs use the model's native coordinate
        system and meters.

        Returns:
            World-space transforms with shape ``[*batch, J, 4, 4]`` in meters.
        """

    def get_rest_pose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
    ) -> dict[str, Float[Array, "..."]]:
        """
        Construct canonical parameter defaults from :attr:`parameter_spec`.

        Args:
            batch_dims: Leading batch dimensions.
            dtype: Optional floating-point dtype.

        Returns:
            Complete model parameters at rest.
        """
        return {name: self._parameter_default(spec, batch_dims, dtype) for name, spec in self.parameter_spec.items()}

    def _parameter_default(
        self,
        spec: ParameterSpec,
        batch_dims: tuple[int, ...],
        dtype: Any | None,
    ) -> Float[Array, "..."]:
        runtime = self.runtime
        reference = self._parameter_reference
        if spec.rotation_type is not None:
            encoded_dims = rotation_ndim(spec.rotation_type)
            rotation_batch = spec.shape[:-encoded_dims]
            like = runtime.zeros(batch_dims, like=reference, dtype=dtype)
            return SO3.identity_as(
                like,
                batch_dims=(*batch_dims, *rotation_batch),
                rotation_type=spec.rotation_type,
                xp=runtime.xp,
            )

        value = runtime.zeros((*batch_dims, *spec.shape), like=reference, dtype=dtype)
        return value if spec.default == 0.0 else value + spec.default

    @property
    @abstractmethod
    def skin_weights(self) -> Float[Array, "V J"]:
        """Skinning weights aligned with the public skeleton. Shape [V, J]."""

    @property
    @abstractmethod
    def rest_vertices(self) -> Float[Array, "V 3"]:
        """Mesh vertices in rest pose. Shape [V, 3]."""

    @property
    def _parameter_reference(self) -> Float[Array, "V 3"]:
        return self.rest_vertices

    @property
    def _skinning_triangles(self) -> Int[Array, "F 3"]:
        return self.faces

    @property
    def _skinning_weights(self) -> Float[Array, "V J"]:
        return self.skin_weights

    @property
    def _corrective_basis(self) -> CorrectiveBasis | None:
        return None

    @property
    def skinning_spec(self) -> SkinningSpec:
        """Static topology, render-rig weights, and optional pose correctives."""
        return SkinningSpec(
            triangles=self._skinning_triangles,
            skinning_weights=self._skinning_weights,
            corrective_basis=self._corrective_basis,
        )

    @abstractmethod
    def forward_vertices(self, *args, **kwargs) -> Float[Array, "*batch V 3"]:
        """
        Compute mesh vertices.

        Signatures vary by model. Outputs use the model's native coordinate
        system and meters.

        Returns:
            Mesh vertices with shape ``[*batch, V, 3]`` in meters.
        """

    @abstractmethod
    def forward_points(self, *args, **kwargs) -> Float[Array, "*batch K 3"]:
        """Compute positions defined by a prepared vertex mapping.

        Signatures vary by model. Outputs use the model's native coordinate
        system and meters.
        """

    def apply_pose_correctives(
        self,
        *,
        identity: SkinningIdentity,
        pose: SkinningPose,
    ) -> Float[Array, "*batch V 3"]:
        """Apply prepared pose correctives to identity-dependent rest vertices."""
        vertices = identity["rest_vertices"]
        coefficients = pose.get("pose_coefficients")
        if coefficients is None:
            return vertices
        basis = self._corrective_basis
        if basis is None:
            raise RuntimeError("Prepared pose has corrective coefficients, but the model has no corrective basis.")
        return vertices + basis.apply(coefficients)

    def prepare_point_regressor(
        self,
        mapping: Float[Array, "K V"],
    ) -> PointRegressor:
        """Preproject a vertex mapping for repeated point forwards.

        For Torch, call this after moving the model to its target device.
        """
        if mapping.ndim != 2 or mapping.shape[0] < 1 or mapping.shape[1] != self.num_vertices:
            raise ValueError(
                f"mapping must have shape [K, {self.num_vertices}] with K >= 1, got {tuple(mapping.shape)}"
            )
        mapping = self._runtime.asarray(mapping, like=self.rest_vertices)
        return point_regression.prepare_point_regressor(
            mapping,
            self._skinning_weights,
            self._corrective_basis,
            runtime=self._runtime,
        )

    def _deform_points(
        self,
        point_regressor: PointRegressor,
        identity: SkinningIdentity,
        pose: SkinningPose,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
        global_translation: Float[Array, "*batch 3"] | None,
    ) -> Float[Array, "*batch K 3"]:
        xp = self._runtime.xp
        rest_points = point_regression.project_rest_points(
            point_regressor,
            identity["rest_vertices"],
            xp=xp,
        )
        points = point_regression.regress_points(point_regressor, rest_points, pose, xp=xp)
        return self._transform_points(points, point_regressor, global_rotation, global_translation)

    def _transform_points(
        self,
        points: Float[Array, "*batch K 3"],
        point_regressor: PointRegressor,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
        global_translation: Float[Array, "*batch 3"] | None,
    ) -> Float[Array, "*batch K 3"]:
        rotation_type = self.parameter_spec["global_rotation"].rotation_type
        if rotation_type is None:
            raise RuntimeError("global_rotation must declare its rotation type")
        points = skinning.apply_global_transform(
            points,
            global_rotation,
            None,
            rotation_type,
            xp=self._runtime.xp,
        )
        if global_translation is not None:
            points = points + global_translation[..., None, :] * point_regressor["weight_sums"][..., None]
        return points

    def _resolve_identity_coefficients(
        self,
        batch_shape: tuple[int, ...],
        /,
        **coefficients: Any | None,
    ) -> tuple[Float[Array, "*batch C"], ...]:
        """Require raw identity coefficients and broadcast them to the pose batch shape."""
        values = [value for value in coefficients.values() if value is not None]
        if len(values) != len(coefficients):
            names = " and ".join(coefficients)
            verb = "is" if len(coefficients) == 1 else "are"
            raise ValueError(f"{names} {verb} required when identity is not provided")
        xp = self._runtime.xp
        return tuple(xp.broadcast_to(value, (*batch_shape, value.shape[-1])) for value in values)

    @staticmethod
    def _validate_identity_arguments(identity: Any | None, **raw_parameters: Any | None) -> None:
        if identity is None:
            return
        conflicts = [name for name, value in raw_parameters.items() if value is not None]
        if conflicts:
            names = ", ".join(conflicts)
            raise ValueError(f"identity cannot be combined with raw identity parameters: {names}")


_JAX_MODELS: set[type] = set()


def _register_jax_model(model_type: type) -> None:
    if model_type in _JAX_MODELS:
        return
    import jax

    jax.tree_util.register_pytree_node(
        model_type,
        _flatten_model,
        partial(_unflatten_model, model_type),
    )
    _JAX_MODELS.add(model_type)


def _flatten_model(model: SkinnedModel) -> tuple[tuple[Any, ...], tuple[Any, ArrayRuntime]]:
    children = tuple(getattr(model, name) for name in model._state_fields)
    return children, (model._config, model._runtime)


def _unflatten_model(
    model_type: type[SkinnedModel],
    auxiliary: tuple[Any, ArrayRuntime],
    children: tuple[Any, ...],
) -> SkinnedModel:
    config, runtime = auxiliary
    model = model_type.__new__(model_type)
    model._runtime = runtime
    model._config = config
    for name, value in zip(model_type._state_fields, children, strict=True):
        setattr(model, name, value)
    return model
