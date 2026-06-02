"""NumPy backend for SOMA model."""

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from ..base import BodyModel
from ..rotations import VALID_ROTATION_TYPES, RotationType
from .io import (
    MODEL_TYPE_SPECS,
    compute_sparse_skin_weights,
    get_model_path,
    load_identity_transfer_data,
    load_model_data,
    simplify_mesh,
)
from body_models.soma.backends import numpy as numpy_backend
from body_models.soma.backends import scipy as scipy_backend
from body_models.soma.backends import core
from body_models.soma import identities
from body_models.soma.identities import numpy as identity_sources
from body_models.soma.constants import SOMA_BODY_PRESETS, SOMA_HAND_PRESETS, SOMA_JOINTS
from body_models.soma.pose import pack_pose, unpack_pose

PathLike = Path | str
SomaIdentity = core.SomaIdentity
SomaPreparedPose = core.SomaPreparedPose

__all__ = ["SOMA"]


class SOMA(BodyModel):
    """SOMA body model with NumPy backend."""

    has_hands = True

    SHAPE_DIM = 128
    NUM_JOINTS = 77
    VALID_MODEL_TYPES = tuple(MODEL_TYPE_SPECS)
    kernels = ("numpy", "scipy")
    JOINTS = SOMA_JOINTS

    _kernel: Any
    weights: Any

    def __init__(
        self,
        model_path: PathLike | None = None,
        *,
        model_type: str = "soma",
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        match_warp: bool = True,
        kernel: Literal["numpy", "scipy"] = "numpy",
    ) -> None:
        """Initialize the SOMA model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            model_type: SOMA identity/model variant to load.
            simplify: Mesh simplification factor to apply while loading.
            rotation_type: Rotation representation expected by pose inputs.
            match_warp: Whether to match Warp backend numerical conventions.
            kernel: Backend kernel used for forward evaluation.
        """
        normalized_model_type = model_type.lower()
        if normalized_model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type: {model_type}. Supported SOMA model types are {', '.join(self.VALID_MODEL_TYPES)}."
            )
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if kernel not in self.kernels:
            raise ValueError(f"Invalid kernel: {kernel}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0 (1.0 = original mesh)")

        self.model_type = normalized_model_type
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self.match_warp = match_warp
        self._kernel = {"numpy": numpy_backend, "scipy": scipy_backend}[kernel]
        resolved_path = get_model_path(model_path)
        data = load_model_data(resolved_path)

        mean_full = data.mean_full
        shapedirs_full = data.shapedirs_full
        faces = data.faces
        skin_weights_full = data.skin_weights_full

        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            mean_active, faces, vertex_map = simplify_mesh(mean_full, faces.astype(int), target_faces)
            shapedirs_active = shapedirs_full[:, vertex_map]
            skin_weights_active = skin_weights_full[vertex_map]
            vertex_map = np.asarray(vertex_map, dtype=np.int64)
        else:
            mean_active = mean_full
            shapedirs_active = shapedirs_full
            skin_weights_active = skin_weights_full
            vertex_map = None

        self.parents = [parent - 1 for parent in data.topology.parents_full[1:]]
        self._joint_names = data.joint_names_full[1:]
        skin_joint_indices_active, skin_joint_weights_active = compute_sparse_skin_weights(skin_weights_active)
        weights = replace(
            data,
            mean_active=np.asarray(mean_active, dtype=np.float32),
            shapedirs_active=np.asarray(shapedirs_active, dtype=np.float32),
            skin_weights_active=np.asarray(skin_weights_active, dtype=np.float32),
            skin_joint_indices_active=skin_joint_indices_active,
            skin_joint_weights_active=skin_joint_weights_active,
            faces=np.asarray(faces, dtype=np.int64),
            vertex_map=vertex_map,
        )
        self.weights = self._kernel.prepare_data(weights)

        spec = MODEL_TYPE_SPECS[self.model_type]
        self.identity_dim = spec.identity_dim
        self.num_scale_params = spec.num_scale_params
        self._default_identity_value = spec.default_identity_value
        self._identity_source = None
        if spec.asset_dir is not None:
            transfer_data = load_identity_transfer_data(resolved_path, self.model_type)
            self._identity_source = identity_sources.create_identity_source(self.model_type, transfer_data)

    @property
    def faces(self) -> Int[np.ndarray, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return self.weights.mean_active.shape[0]

    @property
    def skin_weights(self) -> Float[np.ndarray, "V J"]:
        return self.weights.skin_weights_active[:, 1:]

    @property
    def rest_vertices(self) -> Float[np.ndarray, "V 3"]:
        return self.weights.mean_active * 0.01

    def forward_vertices(
        self,
        body_pose: Float[np.ndarray, "B 23 N"] | Float[np.ndarray, "B 23 3 3"],
        head_pose: Float[np.ndarray, "B 5 N"] | Float[np.ndarray, "B 5 3 3"],
        hand_pose: Float[np.ndarray, "B 48 N"] | Float[np.ndarray, "B 48 3 3"],
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"],
        *,
        shape: Float[np.ndarray, "*batch I"] | None = None,
        scale_params: Float[np.ndarray, "B|1 K"] | None = None,
        identity: SomaIdentity | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        vertex_indices: Any | None = None,
    ) -> Float[np.ndarray, "B V 3"]:
        """Compute posed mesh vertices.

        Args:
            body_pose: Local body joint rotations.
            head_pose: Local head and facial joint rotations.
            hand_pose: Local hand joint rotations.
            global_rotation: Global model rotation.
            shape: Identity coefficients.
            scale_params: Per-part scale parameters.
            identity: Optional output from :meth:`prepare_identity`.
            global_translation: Global model translation.
            vertex_indices: Optional subset of vertices to return.

        Returns:
            Posed vertex positions.
        """
        if identity is None:
            assert shape is not None
            pose = pack_pose(np, global_rotation, body_pose, head_pose, hand_pose)
            batch_shape = tuple(pose.shape[: -(self.num_rot_dims + 1)])
            shape = np.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            if scale_params is not None:
                scale_params = np.broadcast_to(scale_params, (*batch_shape, scale_params.shape[-1]))
            identity = self.prepare_identity(shape, scale_params=scale_params)
        pose = self.prepare_pose(body_pose, head_pose, hand_pose, global_rotation)
        assert "bind_shape_active" in identity
        assert "inverse_world_bind_pose" in identity
        return self._kernel.forward_vertices(
            data=self.weights,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
            bind_shape_active=identity["bind_shape_active"],
            world_bind_pose=identity["world_bind_pose"],
            inverse_world_bind_pose=identity["inverse_world_bind_pose"],
            pose_rot_full=pose["pose_rot_full"],
            xp=np,
        )

    def forward_skeleton(
        self,
        body_pose: Float[np.ndarray, "B 23 N"] | Float[np.ndarray, "B 23 3 3"],
        head_pose: Float[np.ndarray, "B 5 N"] | Float[np.ndarray, "B 5 3 3"],
        hand_pose: Float[np.ndarray, "B 48 N"] | Float[np.ndarray, "B 48 3 3"],
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"],
        *,
        shape: Float[np.ndarray, "*batch I"] | None = None,
        scale_params: Float[np.ndarray, "B|1 K"] | None = None,
        identity: SomaIdentity | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        joint_indices: Any | None = None,
    ) -> Float[np.ndarray, "B 77 4 4"]:
        """Compute posed joint transforms.

        Args:
            body_pose: Local body joint rotations.
            head_pose: Local head and facial joint rotations.
            hand_pose: Local hand joint rotations.
            global_rotation: Global model rotation.
            shape: Identity coefficients.
            scale_params: Per-part scale parameters.
            identity: Optional output from :meth:`prepare_identity`.
            global_translation: Global model translation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
        if identity is None:
            assert shape is not None
            pose = pack_pose(np, global_rotation, body_pose, head_pose, hand_pose)
            batch_shape = tuple(pose.shape[: -(self.num_rot_dims + 1)])
            shape = np.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            if scale_params is not None:
                scale_params = np.broadcast_to(scale_params, (*batch_shape, scale_params.shape[-1]))
            identity = self.prepare_identity(shape, scale_params=scale_params, skip_vertices=True)
        pose = self.prepare_pose(body_pose, head_pose, hand_pose, global_rotation)
        return self._kernel.forward_skeleton(
            data=self.weights,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
            world_bind_pose=identity["world_bind_pose"],
            pose_rot_full=pose["pose_rot_full"],
            xp=np,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype=np.float32,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, np.ndarray]:
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        pose_ref = np.zeros((*batch_dims, self.num_joints, 3), dtype=dtype)
        pose = SO3.identity_as(
            pose_ref,
            batch_dims=(*batch_dims, self.num_joints),
            rotation_type=self.rotation_type,
            xp=np,
        )
        global_rotation, body_pose, head_pose, hand_pose = unpack_pose(np, pose)
        if hands != "default":
            axis_angle = np.asarray(SOMA_HAND_PRESETS[hands], dtype=dtype).reshape(-1, 3)
            axis_angle = np.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
            hand_pose = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=np).copy()
        params = {
            "body_pose": body_pose,
            "head_pose": head_pose,
            "hand_pose": hand_pose,
            "global_rotation": global_rotation,
            "global_translation": np.zeros((*batch_dims, 3), dtype=dtype),
        }
        params["shape"] = np.full(
            (*batch_dims, self.identity_dim),
            self._default_identity_value,
            dtype=dtype,
        )
        if self.num_scale_params is not None:
            params["scale_params"] = np.zeros((*batch_dims, self.num_scale_params), dtype=dtype)
        return params

    def prepare_identity(
        self,
        shape: Float[np.ndarray, "*batch I"],
        *,
        scale_params: Float[np.ndarray, "B|1 K"] | None = None,
        skip_vertices: bool = False,
    ) -> SomaIdentity:
        """Precompute identity-dependent SOMA state for repeated forward passes."""
        if self.num_scale_params is None:
            scale_params = None
        elif scale_params is None:
            scale_params = np.zeros((*shape.shape[:-1], self.num_scale_params), dtype=shape.dtype)
        return self._prepare_identity_from_inputs(shape, scale_params, skip_vertices=skip_vertices)

    def prepare_pose(
        self,
        body_pose: Float[np.ndarray, "B 23 N"] | Float[np.ndarray, "B 23 3 3"],
        head_pose: Float[np.ndarray, "B 5 N"] | Float[np.ndarray, "B 5 3 3"],
        hand_pose: Float[np.ndarray, "B 48 N"] | Float[np.ndarray, "B 48 3 3"],
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"],
        *,
        identity: SomaIdentity | None = None,
    ) -> SomaPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        pose = pack_pose(np, global_rotation, body_pose, head_pose, hand_pose)
        return self._kernel.prepare_pose(self.weights, pose, rotation_type=self.rotation_type, xp=np)

    def _prepare_identity_from_inputs(
        self,
        shape: Float[np.ndarray, "B I"],
        scale_params: Float[np.ndarray, "B K"] | None,
        *,
        skip_vertices: bool = False,
    ) -> SomaIdentity:
        rest_shape_full, rest_shape_active = identities.rest_shapes(
            data=self.weights,
            identity_source=self._identity_source,
            identity=shape,
            scale_params=scale_params,
            xp=np,
        )
        return self._kernel.prepare_identity_from_rest_shape(
            data=self.weights,
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
            match_warp=self.match_warp,
            xp=np,
            skip_vertices=skip_vertices,
        )

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        return params

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        axis_angle = np.asarray(SOMA_BODY_PRESETS["a_pose"], dtype=params["body_pose"].dtype)
        axis_angle = np.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=np).copy()
        return params
