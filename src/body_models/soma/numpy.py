"""NumPy backend for SOMA model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from ..anny.numpy import ANNY
from ..base import BodyModel
from ..mhr.numpy import MHR
from ..rotations import VALID_ROTATION_TYPES, RotationType
from ..smpl.numpy import SMPL
from ..smplx.numpy import SMPLX
from .io import (
    MODEL_TYPE_SPECS,
    compute_kinematic_fronts,
    get_identity_model_path,
    get_model_path,
    load_identity_transfer_data,
    load_model_data,
    simplify_mesh,
)
import body_models.soma.backend.core as soma_base
import body_models.soma.backend.numpy as numpy_kernel
import body_models.soma.backend.scipy as scipy_kernel

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

    mean_full: Float[np.ndarray, "Vf 3"]
    mean_active: Float[np.ndarray, "Va 3"]
    shapedirs_full: Float[np.ndarray, "128 Vf 3"]
    shapedirs_active: Float[np.ndarray, "128 Va 3"]
    eigenvalues: Float[np.ndarray, "128"]
    bind_shape_full: Float[np.ndarray, "Vf 3"]
    bind_pose_world: Float[np.ndarray, "78 4 4"]
    bind_pose_local: Float[np.ndarray, "78 4 4"]
    t_pose_world: Float[np.ndarray, "78 4 4"]
    joint_regressor: Float[np.ndarray, "78 Vf"]
    corrective_bindpose: Float[np.ndarray, "78 3 3"]
    corrective_W1: Float[np.ndarray, "D K"]
    corrective_W2_rows: Int[np.ndarray, "NNZ"]
    corrective_W2_cols: Int[np.ndarray, "NNZ"]
    corrective_W2_values: Float[np.ndarray, "NNZ"]
    _skin_weights_full: Float[np.ndarray, "Vf 78"]
    _skin_weights_active: Float[np.ndarray, "Va 78"]
    _faces: Int[np.ndarray, "F 3"]
    _vertex_map: Int[np.ndarray, "Va"] | None
    _identity_source_tetrahedra: Int[np.ndarray, "Fs 4"] | None
    _identity_face_ids: Int[np.ndarray, "Vt"] | None
    _identity_bary_coords: Float[np.ndarray, "Vt 4"] | None
    _identity_unknown_ids: Int[np.ndarray, "U"] | None
    _identity_anchor_ids: Int[np.ndarray, "A"] | None
    _identity_solve_matrix: Float[np.ndarray, "U U"] | None
    _identity_anchor_matrix: Float[np.ndarray, "U A"] | None
    _identity_rhs_base: Float[np.ndarray, "U 3"] | None
    _identity_internal_to_source_rotation: Float[np.ndarray, "3 3"]
    _identity_internal_to_source_translation: Float[np.ndarray, "3"]
    _identity_source_to_soma_rotation: Float[np.ndarray, "3 3"]
    _identity_model: object | None
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
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"

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
            self._vertex_map = np.asarray(vertex_map, dtype=np.int64)
        else:
            mean_active = mean_full
            shapedirs_active = shapedirs_full
            skin_weights_active = skin_weights_full
            self._vertex_map = None

        self.mean_full = mean_full
        self.mean_active = np.asarray(mean_active, dtype=np.float32)
        self.shapedirs_full = shapedirs_full
        self.shapedirs_active = np.asarray(shapedirs_active, dtype=np.float32)
        self.eigenvalues = data.eigenvalues
        self.bind_shape_full = data.bind_shape_full
        self.bind_pose_world = data.bind_pose_world
        self.bind_pose_local = data.bind_pose_local
        self.t_pose_world = data.t_pose_world
        self.joint_regressor = data.joint_regressor
        self.corrective_bindpose = np.asarray(data.correctives.corrective_bindpose, dtype=np.float32)
        self.corrective_W1 = np.asarray(data.correctives.corrective_W1, dtype=np.float32)
        self.corrective_W2_rows = np.asarray(data.correctives.corrective_W2_rows, dtype=np.int64)
        self.corrective_W2_cols = np.asarray(data.correctives.corrective_W2_cols, dtype=np.int64)
        self.corrective_W2_values = np.asarray(data.correctives.corrective_W2_values, dtype=np.float32)
        self._corrective_use_tanh = data.corrective_use_tanh
        self._skin_weights_full = skin_weights_full
        self._skin_weights_active = np.asarray(skin_weights_active, dtype=np.float32)
        self._faces = np.asarray(faces, dtype=np.int64)
        self._identity_internal_to_source_rotation = np.eye(3, dtype=np.float32)
        self._identity_internal_to_source_translation = np.zeros(3, dtype=np.float32)
        self._identity_source_to_soma_rotation = np.eye(3, dtype=np.float32)

        self.parents = list(data.parents)
        self._parents_full = data.topology.parents_full
        self._joint_children_full = data.topology.joint_children_full
        self._skinned_vertex_indices_full = data.topology.skinned_vertex_indices_full
        self._parents_full_index = np.asarray(self._parents_full, dtype=np.int64)
        self._joint_children_indices_full = data.topology.joint_children_indices_full
        self._skinned_vertex_indices_full_index = data.topology.skinned_vertex_indices_full_index
        self._kinematic_fronts_full = compute_kinematic_fronts(self._parents_full)
        self._joint_names = list(data.joint_names)
        self.model_weights = self._kernel.prepare_data(
            data,
            mean_active=self.mean_active,
            shapedirs_active=self.shapedirs_active,
            skin_weights_active=self._skin_weights_active,
            faces=self._faces,
            vertex_map=self._vertex_map,
            parents_full=self._parents_full,
            parents_full_index=self._parents_full_index,
        )

        spec = MODEL_TYPE_SPECS[self.model_type]
        self.identity_dim = spec.identity_dim
        self.num_scale_params = spec.num_scale_params
        self._default_identity_value = spec.default_identity_value
        self._identity_source_scale = spec.source_scale
        self._identity_output_scale = spec.output_scale
        self._identity_model = None
        self._identity_source_tetrahedra = None
        self._identity_face_ids = None
        self._identity_bary_coords = None
        self._identity_unknown_ids = None
        self._identity_anchor_ids = None
        self._identity_solve_matrix = None
        self._identity_anchor_matrix = None
        self._identity_rhs_base = None

        if spec.asset_dir is None:
            return

        transfer_data = load_identity_transfer_data(resolved_path, self.model_type)
        self._identity_source_tetrahedra = np.asarray(transfer_data["source_tetrahedra"], dtype=np.int64)
        self._identity_face_ids = np.asarray(transfer_data["face_ids"], dtype=np.int64)
        self._identity_bary_coords = np.asarray(transfer_data["bary_coords"], dtype=np.float32)
        self._identity_unknown_ids = np.asarray(transfer_data["unknown_ids"], dtype=np.int64)
        self._identity_anchor_ids = np.asarray(transfer_data["anchor_ids"], dtype=np.int64)
        self._identity_solve_matrix = np.asarray(transfer_data["solve_matrix"], dtype=np.float32)
        self._identity_anchor_matrix = np.asarray(transfer_data["anchor_matrix"], dtype=np.float32)
        self._identity_rhs_base = np.asarray(transfer_data["rhs_base"], dtype=np.float32)
        self._identity_model = {
            "mhr": self._init_mhr_identity_backend,
            "anny": self._init_anny_identity_backend,
            "smpl": self._init_linear_identity_backend,
            "smplx": self._init_linear_identity_backend,
        }[self.model_type](transfer_data)

    @property
    def faces(self) -> Int[np.ndarray, "F 3"]:
        return self._faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return self.mean_active.shape[0]

    @property
    def skin_weights(self) -> Float[np.ndarray, "V J"]:
        return self._skin_weights_active[:, 1:]

    @property
    def rest_vertices(self) -> Float[np.ndarray, "V 3"]:
        return self.mean_active * 0.01

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
            corrective_use_tanh=self._corrective_use_tanh,
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
        params["identity"] = np.full((1, self.identity_dim), self._default_identity_value, dtype=dtype)
        if self.num_scale_params is not None:
            params["scale_params"] = np.zeros((1, self.num_scale_params), dtype=dtype)
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
            model_type=self.model_type,
            identity_model=self._identity_model,
            identity=identity,
            scale_params=scale_params,
            batch_size=batch_size,
            identity_dim=self.identity_dim,
            default_identity_value=self._default_identity_value,
            num_scale_params=self.num_scale_params,
            identity_internal_to_source_rotation=self._identity_internal_to_source_rotation,
            identity_internal_to_source_translation=self._identity_internal_to_source_translation,
            identity_source_to_soma_rotation=self._identity_source_to_soma_rotation,
            identity_source_scale=self._identity_source_scale,
            identity_output_scale=self._identity_output_scale,
            identity_source_tetrahedra=self._identity_source_tetrahedra,
            identity_face_ids=self._identity_face_ids,
            identity_bary_coords=self._identity_bary_coords,
            identity_unknown_ids=self._identity_unknown_ids,
            identity_anchor_ids=self._identity_anchor_ids,
            identity_solve_matrix=self._identity_solve_matrix,
            identity_anchor_matrix=self._identity_anchor_matrix,
            identity_rhs_base=self._identity_rhs_base,
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

    def _init_mhr_identity_backend(self, _transfer_data: dict[str, np.ndarray]) -> MHR:
        return MHR(model_path=get_identity_model_path("mhr"), simplify=1.0)

    def _init_anny_identity_backend(self, transfer_data: dict[str, np.ndarray]) -> object:
        identity_model = ANNY(
            model_path=get_identity_model_path("anny"),
            all_phenotypes=False,
            simplify=1.0,
        )
        source_vertices = np.asarray(transfer_data["source_vertices"], dtype=np.float32)
        rotation, translation = self._kernel.fit_rigid_transform(
            identity_model.template_vertices,
            source_vertices,
            xp=np,
        )
        self._identity_internal_to_source_rotation = rotation.astype(np.float32, copy=False)
        self._identity_internal_to_source_translation = translation.astype(np.float32, copy=False)
        self._identity_source_to_soma_rotation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )
        return soma_base.AnnyIdentityData(
            template_vertices=identity_model.template_vertices,
            blendshapes=identity_model.blendshapes,
            phenotype_mask=identity_model.phenotype_mask,
            anchors=identity_model._anchors,
        )

    def _init_linear_identity_backend(self, _transfer_data: dict[str, np.ndarray]) -> object:
        linear_model_cls = {"smpl": SMPL, "smplx": SMPLX}[self.model_type]
        identity_model = linear_model_cls(
            model_path=get_identity_model_path(self.model_type),
            simplify=1.0,
        )
        return soma_base.LinearIdentityData(
            mean=identity_model.v_template_full,
            shapedirs=identity_model.shapedirs_full,
        )
