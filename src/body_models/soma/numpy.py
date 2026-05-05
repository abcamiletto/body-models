"""NumPy backend for SOMA model."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from ..base import BodyModel
from ..rotations import VALID_ROTATION_TYPES, RotationType
from .io import (
    MODEL_TYPE_SPECS,
    get_model_path,
    load_identity_transfer_data,
    load_model_data,
    simplify_mesh,
)
import body_models.soma.backend.numpy as numpy_kernel
import body_models.soma.backend.scipy as scipy_kernel
from body_models.soma import identities

PathLike = Path | str
KernelBackend = Literal["numpy", "scipy"]

__all__ = ["SOMA", "SOMAIdentity"]


@dataclass
class SOMAIdentity:
    rest_shape_full: Float[np.ndarray, "1 Vf 3"]
    rest_shape_active: Float[np.ndarray, "1 Va 3"]
    world_bind_pose_fit: Float[np.ndarray, "1 Jf 4 4"]


class SOMA(BodyModel):
    """SOMA body model with NumPy backend."""

    SHAPE_DIM = 128
    NUM_JOINTS = 77
    VALID_MODEL_TYPES = tuple(MODEL_TYPE_SPECS)

    identity_backend: identities.IdentityBackend
    _kernel: Any
    model_weights: Any

    def __init__(
        self,
        model_path: PathLike | None = None,
        *,
        model_type: str = "soma",
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        match_warp: bool = True,
        backend: KernelBackend = "numpy",
    ) -> None:
        normalized_model_type = model_type.lower()
        if normalized_model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type: {model_type}. Supported SOMA model types are {', '.join(self.VALID_MODEL_TYPES)}."
            )
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0 (1.0 = original mesh)")

        self.model_type = normalized_model_type
        self.rotation_type = rotation_type
        self.match_warp = match_warp
        self._kernel = {"numpy": numpy_kernel, "scipy": scipy_kernel}[backend]
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
        weights = replace(
            data,
            mean_active=np.asarray(mean_active, dtype=np.float32),
            shapedirs_active=np.asarray(shapedirs_active, dtype=np.float32),
            skin_weights_active=np.asarray(skin_weights_active, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int64),
            vertex_map=vertex_map,
        )
        self.model_weights = self._kernel.prepare_data(weights)

        spec = MODEL_TYPE_SPECS[self.model_type]
        transfer_data = None
        if spec.asset_dir is not None:
            transfer_data = load_identity_transfer_data(resolved_path, self.model_type)
        identity_backend = identities.load(self.model_type, spec, transfer_data)
        self.identity_backend = self._kernel.prepare_identity_backend(identity_backend)

    @property
    def faces(self) -> Int[np.ndarray, "F 3"]:
        return self.model_weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return self.model_weights.mean_active.shape[0]

    @property
    def skin_weights(self) -> Float[np.ndarray, "V J"]:
        return self.model_weights.skin_weights_active[:, 1:]

    @property
    def rest_vertices(self) -> Float[np.ndarray, "V 3"]:
        return self.model_weights.mean_active * 0.01

    def forward_vertices(
        self,
        pose: Float[np.ndarray, "B 77 N"] | Float[np.ndarray, "B 77 3 3"],
        *,
        identity: Float[np.ndarray, "B|1 I"] | None = None,
        scale_params: Float[np.ndarray, "B|1 K"] | None = None,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        vertex_indices=None,
        apply_correctives: bool = True,
        prepared_identity: SOMAIdentity | None = None,
    ) -> Float[np.ndarray, "B V 3"]:
        identity_state = self._prepare_or_use_identity(
            identity=identity,
            scale_params=scale_params,
            dtype=pose.dtype,
            prepared_identity=prepared_identity,
        )
        return self._kernel.forward_vertices(
            data=self.model_weights,
            identity=None,
            pose=pose,
            rest_shape_full=identity_state.rest_shape_full,
            rest_shape_active=identity_state.rest_shape_active,
            world_bind_pose_fit=identity_state.world_bind_pose_fit,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
            match_warp=self.match_warp,
            xp=np,
        )

    def forward_skeleton(
        self,
        pose: Float[np.ndarray, "B 77 N"] | Float[np.ndarray, "B 77 3 3"],
        *,
        identity: Float[np.ndarray, "B|1 I"] | None = None,
        scale_params: Float[np.ndarray, "B|1 K"] | None = None,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        joint_indices=None,
        apply_correctives: bool = True,
        prepared_identity: SOMAIdentity | None = None,
    ) -> Float[np.ndarray, "B 77 4 4"]:
        identity_state = self._prepare_or_use_identity(
            identity=identity,
            scale_params=scale_params,
            dtype=pose.dtype,
            prepared_identity=prepared_identity,
        )
        return self._kernel.forward_skeleton(
            data=self.model_weights,
            identity=None,
            pose=pose,
            rest_shape_full=identity_state.rest_shape_full,
            world_bind_pose_fit=identity_state.world_bind_pose_fit,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
            match_warp=self.match_warp,
            xp=np,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=np.float32) -> dict[str, np.ndarray]:
        pose_ref = np.zeros((batch_size, self.num_joints, 3), dtype=dtype)
        rot_ref = np.zeros((batch_size, 3), dtype=dtype)
        params = {
            "pose": SO3.identity_as(
                pose_ref,
                batch_dims=(batch_size, self.num_joints),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "global_rotation": SO3.identity_as(
                rot_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "global_translation": np.zeros((batch_size, 3), dtype=dtype),
        }
        params["identity"] = np.full(
            (1, self.identity_backend.identity_dim),
            self.identity_backend.default_identity_value,
            dtype=dtype,
        )
        if self.identity_backend.num_scale_params is not None:
            params["scale_params"] = np.zeros((1, self.identity_backend.num_scale_params), dtype=dtype)
        return params

    def prepare_identity(
        self,
        *,
        identity: Float[np.ndarray, "B|1 I"] | None = None,
        scale_params: Float[np.ndarray, "B|1 K"] | None = None,
        dtype=np.float32,
    ) -> SOMAIdentity:
        if identity is not None:
            dtype = identity.dtype
        ref = np.empty((1, 1), dtype=dtype)
        batch_size = 1 if identity is None else identity.shape[0]
        _identity, rest_shape_full, rest_shape_active, world_bind_pose_fit = self._kernel.prepare_identity(
            data=self.model_weights,
            identity_backend=self.identity_backend,
            identity=identity,
            scale_params=scale_params,
            batch_size=batch_size,
            match_warp=self.match_warp,
            ref=ref,
            xp=np,
        )
        return SOMAIdentity(rest_shape_full, rest_shape_active, world_bind_pose_fit)

    def _prepare_or_use_identity(
        self,
        *,
        identity: Float[np.ndarray, "B|1 I"] | None,
        scale_params: Float[np.ndarray, "B|1 K"] | None,
        dtype: np.dtype,
        prepared_identity: SOMAIdentity | None,
    ) -> SOMAIdentity:
        if prepared_identity is not None:
            return prepared_identity
        return self.prepare_identity(identity=identity, scale_params=scale_params, dtype=dtype)
