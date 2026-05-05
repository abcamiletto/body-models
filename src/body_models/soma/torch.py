"""PyTorch backend for SOMA model."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from ..anny.torch import ANNY
from ..base import BodyModel
from ..mhr.torch import MHR
from ..rotations import VALID_ROTATION_TYPES
from ..smpl.torch import SMPL
from ..smplx.torch import SMPLX
from . import core, identities
from .io import (
    MODEL_TYPE_SPECS,
    SomaIdentityTransfer,
    compute_kinematic_fronts,
    get_identity_model_path,
    get_model_path,
    load_identity_transfer_data,
    load_model_data,
    simplify_mesh,
)

PathLike = Path | str

__all__ = ["SOMA"]


class SOMA(BodyModel, nn.Module):
    """SOMA body model with PyTorch backend."""

    SHAPE_DIM = 128
    NUM_JOINTS = 77
    VALID_MODEL_TYPES = tuple(MODEL_TYPE_SPECS)

    mean_full: Float[Tensor, "Vf 3"]
    mean_active: Float[Tensor, "Va 3"]
    shapedirs_full: Float[Tensor, "128 Vf 3"]
    shapedirs_active: Float[Tensor, "128 Va 3"]
    eigenvalues: Float[Tensor, "128"]
    bind_shape_full: Float[Tensor, "Vf 3"]
    bind_pose_world: Float[Tensor, "78 4 4"]
    bind_pose_local: Float[Tensor, "78 4 4"]
    t_pose_world: Float[Tensor, "78 4 4"]
    joint_regressor: Float[Tensor, "78 Vf"]
    corrective_bindpose: Float[Tensor, "78 3 3"]
    corrective_W1: Float[Tensor, "D K"]
    corrective_W2_rows: Int[Tensor, "NNZ"]
    corrective_W2_cols: Int[Tensor, "NNZ"]
    corrective_W2_values: Float[Tensor, "NNZ"]
    _skin_weights_full: Float[Tensor, "Vf 78"]
    _skin_weights_active: Float[Tensor, "Va 78"]
    _faces: Int[Tensor, "F 3"]
    _vertex_map: Int[Tensor, "Va"] | None
    _identity_source_vertices: Float[Tensor, "Vs 3"]
    _identity_source_tetrahedra: Int[Tensor, "Fs 4"]
    _identity_face_ids: Int[Tensor, "Vt"]
    _identity_bary_coords: Float[Tensor, "Vt 4"]
    _identity_unknown_ids: Int[Tensor, "U"]
    _identity_anchor_ids: Int[Tensor, "A"]
    _identity_solve_matrix: Float[Tensor, "U U"]
    _identity_anchor_matrix: Float[Tensor, "U A"]
    _identity_rhs_base: Float[Tensor, "U 3"]
    _identity_internal_to_source_rotation: Float[Tensor, "3 3"]
    _identity_internal_to_source_translation: Float[Tensor, "3"]
    _identity_source_to_soma_rotation: Float[Tensor, "3 3"]
    _identity_anny_model: ANNY
    _identity_mhr_model: MHR
    _identity_linear_model: SMPL | SMPLX

    def __init__(
        self,
        model_path: PathLike | None = None,
        *,
        model_type: str = "soma",
        simplify: float = 1.0,
        rotation_type: core.RotationType = "axis_angle",
        match_warp: bool = True,
    ) -> None:
        normalized_model_type = model_type.lower()
        if normalized_model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type: {model_type}. Supported SOMA model types are {', '.join(self.VALID_MODEL_TYPES)}."
            )
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"
        super().__init__()

        self.model_type = normalized_model_type
        self.rotation_type = rotation_type
        self.match_warp = match_warp
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
            self.register_buffer("_vertex_map", torch.as_tensor(np.asarray(vertex_map, dtype=np.int64)))
        else:
            mean_active = mean_full
            shapedirs_active = shapedirs_full
            skin_weights_active = skin_weights_full
            self._vertex_map = None

        self.register_buffer("mean_full", torch.as_tensor(mean_full))
        self.register_buffer("mean_active", torch.as_tensor(mean_active))
        self.register_buffer("shapedirs_full", torch.as_tensor(shapedirs_full))
        self.register_buffer("shapedirs_active", torch.as_tensor(shapedirs_active))
        self.register_buffer("eigenvalues", torch.as_tensor(data.eigenvalues))
        self.register_buffer("bind_shape_full", torch.as_tensor(data.bind_shape_full))
        self.register_buffer("bind_pose_world", torch.as_tensor(data.bind_pose_world))
        self.register_buffer("bind_pose_local", torch.as_tensor(data.bind_pose_local))
        self.register_buffer("t_pose_world", torch.as_tensor(data.t_pose_world))
        self.register_buffer("joint_regressor", torch.as_tensor(data.joint_regressor))
        self.register_buffer("corrective_bindpose", torch.as_tensor(data.correctives.corrective_bindpose))
        self.register_buffer("corrective_W1", torch.as_tensor(data.correctives.corrective_W1))
        self.register_buffer(
            "corrective_W2_rows",
            torch.as_tensor(data.correctives.corrective_W2_rows, dtype=torch.int64),
        )
        self.register_buffer(
            "corrective_W2_cols",
            torch.as_tensor(data.correctives.corrective_W2_cols, dtype=torch.int64),
        )
        self.register_buffer("corrective_W2_values", torch.as_tensor(data.correctives.corrective_W2_values))
        self.register_buffer("_skin_weights_full", torch.as_tensor(skin_weights_full))
        self.register_buffer("_skin_weights_active", torch.as_tensor(skin_weights_active))
        self.register_buffer("_faces", torch.as_tensor(np.asarray(faces, dtype=np.int64)))
        self.register_buffer("_identity_internal_to_source_rotation", torch.eye(3, dtype=self.mean_full.dtype))
        self.register_buffer("_identity_internal_to_source_translation", torch.zeros(3, dtype=self.mean_full.dtype))
        self.register_buffer("_identity_source_to_soma_rotation", torch.eye(3, dtype=self.mean_full.dtype))

        self._corrective_use_tanh = False
        self.parents = [parent - 1 for parent in data.topology.parents_full[1:]]
        self._parents_full = data.topology.parents_full
        self._joint_children_full = data.topology.joint_children_full
        self._skinned_vertex_indices_full = data.topology.skinned_vertex_indices_full
        self.register_buffer("_parents_full_index", torch.as_tensor(self._parents_full, dtype=torch.int64))
        self.register_buffer("_joint_children_indices_full", torch.as_tensor(data.topology.joint_children_indices_full))
        self.register_buffer(
            "_skinned_vertex_indices_full_index",
            torch.as_tensor(data.topology.skinned_vertex_indices_full_index),
        )
        self._kinematic_fronts_full = compute_kinematic_fronts(self._parents_full)
        self._joint_names = data.joint_names_full[1:]

        spec = MODEL_TYPE_SPECS[self.model_type]
        self.identity_dim = spec.identity_dim
        self.num_scale_params = spec.num_scale_params
        self._default_identity_value = spec.default_identity_value
        self._identity_source_scale = spec.source_scale
        self._identity_output_scale = spec.output_scale

        if spec.asset_dir is None:
            return

        transfer_data = load_identity_transfer_data(resolved_path, self.model_type)
        self.register_buffer("_identity_source_vertices", torch.as_tensor(transfer_data.source_vertices))
        self.register_buffer(
            "_identity_source_tetrahedra",
            torch.as_tensor(transfer_data.source_tetrahedra, dtype=torch.int64),
        )
        self.register_buffer("_identity_face_ids", torch.as_tensor(transfer_data.face_ids, dtype=torch.int64))
        self.register_buffer("_identity_bary_coords", torch.as_tensor(transfer_data.bary_coords))
        self.register_buffer("_identity_unknown_ids", torch.as_tensor(transfer_data.unknown_ids, dtype=torch.int64))
        self.register_buffer("_identity_anchor_ids", torch.as_tensor(transfer_data.anchor_ids, dtype=torch.int64))
        self.register_buffer("_identity_solve_matrix", torch.as_tensor(transfer_data.solve_matrix))
        self.register_buffer("_identity_anchor_matrix", torch.as_tensor(transfer_data.anchor_matrix))
        self.register_buffer("_identity_rhs_base", torch.as_tensor(transfer_data.rhs_base))
        {
            "mhr": self._init_mhr_identity_backend,
            "anny": self._init_anny_identity_backend,
            "smpl": self._init_linear_identity_backend,
            "smplx": self._init_linear_identity_backend,
        }[self.model_type](transfer_data)

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self._faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return int(self.mean_active.shape[0])

    @property
    def skin_weights(self) -> Float[Tensor, "V J"]:
        return self._skin_weights_active[:, 1:]

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.mean_active * 0.01

    def forward_vertices(
        self,
        pose: Float[Tensor, "B 77 N"] | Float[Tensor, "B 77 3 3"],
        *,
        identity: Float[Tensor, "B|1 I"] | None = None,
        scale_params: Float[Tensor, "B|1 K"] | None = None,
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        vertex_indices=None,
        apply_correctives: bool = True,
    ) -> Float[Tensor, "B V 3"]:
        identity, rest_shape_full, rest_shape_active = self._get_rest_shape(
            identity=identity,
            scale_params=scale_params,
            ref=pose,
        )
        return core.forward_vertices(
            mean_full=self.mean_full,
            mean_active=self.mean_active,
            shapedirs_full=self.shapedirs_full,
            shapedirs_active=self.shapedirs_active,
            eigenvalues=self.eigenvalues,
            bind_shape_full=self.bind_shape_full,
            skin_weights_active=self._skin_weights_active,
            bind_pose_world=self.bind_pose_world,
            bind_pose_local=self.bind_pose_local,
            t_pose_world=self.t_pose_world,
            joint_regressor=self.joint_regressor,
            joint_children_full=self._joint_children_full,
            joint_children_indices_full=self._joint_children_indices_full,
            skinned_vertex_indices_full=self._skinned_vertex_indices_full,
            skinned_vertex_indices_full_index=self._skinned_vertex_indices_full_index,
            kinematic_fronts_full=self._kinematic_fronts_full,
            parents_full=self._parents_full,
            parents_full_index=self._parents_full_index,
            identity=identity,
            pose=pose,
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            vertex_map=self._vertex_map,
            corrective_bindpose=self.corrective_bindpose,
            corrective_W1=self.corrective_W1,
            corrective_W2_rows=self.corrective_W2_rows,
            corrective_W2_cols=self.corrective_W2_cols,
            corrective_W2_values=self.corrective_W2_values,
            corrective_use_tanh=self._corrective_use_tanh,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
            match_warp=self.match_warp,
            xp=torch,
        )

    def forward_skeleton(
        self,
        pose: Float[Tensor, "B 77 N"] | Float[Tensor, "B 77 3 3"],
        *,
        identity: Float[Tensor, "B|1 I"] | None = None,
        scale_params: Float[Tensor, "B|1 K"] | None = None,
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        joint_indices=None,
        apply_correctives: bool = True,
    ) -> Float[Tensor, "B 77 4 4"]:
        identity, rest_shape_full, _rest_shape_active = self._get_rest_shape(
            identity=identity,
            scale_params=scale_params,
            ref=pose,
        )
        return core.forward_skeleton(
            mean_full=self.mean_full,
            shapedirs_full=self.shapedirs_full,
            eigenvalues=self.eigenvalues,
            bind_shape_full=self.bind_shape_full,
            skin_weights_full=self._skin_weights_full,
            bind_pose_world=self.bind_pose_world,
            bind_pose_local=self.bind_pose_local,
            t_pose_world=self.t_pose_world,
            joint_regressor=self.joint_regressor,
            joint_children_full=self._joint_children_full,
            joint_children_indices_full=self._joint_children_indices_full,
            skinned_vertex_indices_full=self._skinned_vertex_indices_full,
            skinned_vertex_indices_full_index=self._skinned_vertex_indices_full_index,
            kinematic_fronts_full=self._kinematic_fronts_full,
            parents_full=self._parents_full,
            parents_full_index=self._parents_full_index,
            identity=identity,
            pose=pose,
            rest_shape_full=rest_shape_full,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
            match_warp=self.match_warp,
            xp=torch,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        device = self.mean_active.device
        pose_ref = torch.zeros((batch_size, self.num_joints, 3), device=device, dtype=dtype)
        rot_ref = torch.zeros((batch_size, 3), device=device, dtype=dtype)
        params = {
            "pose": SO3.identity_as(
                pose_ref,
                batch_dims=(batch_size, self.num_joints),
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "global_rotation": SO3.identity_as(
                rot_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "global_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }
        params["identity"] = torch.full(
            (1, self.identity_dim),
            self._default_identity_value,
            device=device,
            dtype=dtype,
        )
        if self.num_scale_params is not None:
            params["scale_params"] = torch.zeros((1, self.num_scale_params), device=device, dtype=dtype)
        return params

    def _get_rest_shape(
        self,
        *,
        identity: Float[Tensor, "B|1 I"] | None,
        scale_params: Float[Tensor, "B|1 K"] | None,
        ref: Float[Tensor, "B ..."],
    ) -> tuple[Float[Tensor, "B|1 I"] | None, Float[Tensor, "B V 3"] | None, Float[Tensor, "B V 3"] | None]:
        if identity is None:
            identity = torch.full(
                (1, self.identity_dim), self._default_identity_value, device=ref.device, dtype=ref.dtype
            )
        identity, scale_params = core.resolve_identity_inputs(
            identity=identity,
            scale_params=scale_params,
            batch_size=ref.shape[0],
            identity_dim=self.identity_dim,
            num_scale_params=self.num_scale_params,
            ref=ref,
            xp=torch,
        )
        if self.model_type == "soma":
            return identity, None, None

        if self.model_type == "mhr":
            assert self.num_scale_params is not None
            source_shape = core.mhr_identity_shape(
                model=self._identity_mhr_model,
                identity=identity,
                scale_params=scale_params,
                num_scale_params=self.num_scale_params,
                xp=torch,
            )
        elif self.model_type == "anny":
            source_shape = core.anny_identity_shape(
                template_vertices=self._identity_anny_model.template_vertices,
                blendshapes=self._identity_anny_model.blendshapes,
                phenotype_mask=self._identity_anny_model.phenotype_mask,
                anchors=self._identity_anny_model._get_anchors_dict(),
                identity=identity,
                xp=torch,
            )
        else:
            source_shape = identities.linear_identity_shape(
                mean=self._identity_linear_model.v_template_full,
                shapedirs=self._identity_linear_model.shapedirs_full,
                identity=identity,
                xp=torch,
            )
        rest_shape, rest_shape_active = identities.transfer_shape(
            source_shape,
            transfer=self._current_identity_transfer(),
            vertex_map=self._vertex_map,
            xp=torch,
        )
        return None, rest_shape, rest_shape_active

    def _current_identity_transfer(self) -> identities.IdentityTransfer:
        return identities.IdentityTransfer(
            source_vertices=self._identity_source_vertices,
            source_tetrahedra=self._identity_source_tetrahedra,
            face_ids=self._identity_face_ids,
            bary_coords=self._identity_bary_coords,
            unknown_ids=self._identity_unknown_ids,
            anchor_ids=self._identity_anchor_ids,
            solve_matrix=self._identity_solve_matrix,
            anchor_matrix=self._identity_anchor_matrix,
            rhs_base=self._identity_rhs_base,
            internal_to_source_rotation=self._identity_internal_to_source_rotation,
            internal_to_source_translation=self._identity_internal_to_source_translation,
            source_to_soma_rotation=self._identity_source_to_soma_rotation,
            source_scale=self._identity_source_scale,
            output_scale=self._identity_output_scale,
        )

    def _init_mhr_identity_backend(self, _transfer_data: SomaIdentityTransfer) -> None:
        self._identity_mhr_model = MHR(model_path=get_identity_model_path("mhr"), simplify=1.0)

    def _init_anny_identity_backend(self, transfer_data: SomaIdentityTransfer) -> None:
        self._identity_anny_model = ANNY(
            model_path=get_identity_model_path("anny"),
            all_phenotypes=False,
            simplify=1.0,
        )
        source_vertices = torch.as_tensor(transfer_data.source_vertices, dtype=self.mean_full.dtype)
        rotation, translation = core.fit_rigid_transform(
            self._identity_anny_model.template_vertices,
            source_vertices,
            xp=torch,
        )
        self._identity_internal_to_source_rotation = rotation
        self._identity_internal_to_source_translation = translation
        self._identity_source_to_soma_rotation = torch.as_tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=self.mean_full.dtype,
        )

    def _init_linear_identity_backend(self, _transfer_data: SomaIdentityTransfer) -> None:
        linear_model_cls = {"smpl": SMPL, "smplx": SMPLX}[self.model_type]
        self._identity_linear_model = linear_model_cls(
            model_path=get_identity_model_path(self.model_type),
            simplify=1.0,
        )
