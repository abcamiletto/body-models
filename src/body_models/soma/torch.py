"""PyTorch backend for SOMA model."""

from dataclasses import replace
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
from ..rotations import VALID_ROTATION_TYPES, RotationType
from ..smpl.torch import SMPL
from ..smplx.torch import SMPLX
from .io import (
    MODEL_TYPE_SPECS,
    get_identity_model_path,
    get_model_path,
    load_identity_transfer_data,
    load_model_data,
    simplify_mesh,
)
import body_models.soma.backend.core as soma_base
import body_models.soma.backend.torch as core

PathLike = Path | str

__all__ = ["SOMA"]


class SOMA(BodyModel, nn.Module):
    """SOMA body model with PyTorch backend."""

    SHAPE_DIM = 128
    NUM_JOINTS = 77
    VALID_MODEL_TYPES = tuple(MODEL_TYPE_SPECS)

    model_weights: core.SomaTorchWeights
    _identity_source_tetrahedra: Int[Tensor, "Fs 4"] | None
    _identity_face_ids: Int[Tensor, "Vt"] | None
    _identity_bary_coords: Float[Tensor, "Vt 4"] | None
    _identity_unknown_ids: Int[Tensor, "U"] | None
    _identity_anchor_ids: Int[Tensor, "A"] | None
    _identity_solve_matrix: Float[Tensor, "U U"] | None
    _identity_anchor_matrix: Float[Tensor, "U A"] | None
    _identity_rhs_base: Float[Tensor, "U 3"] | None
    _identity_internal_to_source_rotation: Float[Tensor, "3 3"]
    _identity_internal_to_source_translation: Float[Tensor, "3"]
    _identity_source_to_soma_rotation: Float[Tensor, "3 3"]
    _identity_model: object

    def __init__(
        self,
        model_path: PathLike | None = None,
        *,
        model_type: str = "soma",
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
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
            vertex_map = np.asarray(vertex_map, dtype=np.int64)
        else:
            mean_active = mean_full
            shapedirs_active = shapedirs_full
            skin_weights_active = skin_weights_full
            vertex_map = None

        weights = replace(
            data,
            mean_active=np.asarray(mean_active, dtype=np.float32),
            shapedirs_active=np.asarray(shapedirs_active, dtype=np.float32),
            skin_weights_active=np.asarray(skin_weights_active, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int64),
            vertex_map=vertex_map,
        )
        self.model_weights = core.prepare_data(weights)
        dtype = self.model_weights.mean_full.dtype
        self.register_buffer("_identity_internal_to_source_rotation", torch.eye(3, dtype=dtype))
        self.register_buffer("_identity_internal_to_source_translation", torch.zeros(3, dtype=dtype))
        self.register_buffer("_identity_source_to_soma_rotation", torch.eye(3, dtype=dtype))
        self.parents = [parent - 1 for parent in data.topology.parents_full[1:]]
        self._joint_names = data.joint_names_full[1:]

        spec = MODEL_TYPE_SPECS[self.model_type]
        self.identity_dim = spec.identity_dim
        self.num_scale_params = spec.num_scale_params
        self._default_identity_value = spec.default_identity_value
        self._identity_source_scale = spec.source_scale
        self._identity_output_scale = spec.output_scale
        self._identity_model = None
        self.register_buffer("_identity_source_tetrahedra", None)
        self.register_buffer("_identity_face_ids", None)
        self.register_buffer("_identity_bary_coords", None)
        self.register_buffer("_identity_unknown_ids", None)
        self.register_buffer("_identity_anchor_ids", None)
        self.register_buffer("_identity_solve_matrix", None)
        self.register_buffer("_identity_anchor_matrix", None)
        self.register_buffer("_identity_rhs_base", None)

        if spec.asset_dir is None:
            return

        transfer_data = load_identity_transfer_data(resolved_path, self.model_type)
        self.register_buffer(
            "_identity_source_tetrahedra",
            torch.as_tensor(transfer_data["source_tetrahedra"], dtype=torch.int64),
        )
        self.register_buffer("_identity_face_ids", torch.as_tensor(transfer_data["face_ids"], dtype=torch.int64))
        self.register_buffer("_identity_bary_coords", torch.as_tensor(transfer_data["bary_coords"]))
        self.register_buffer("_identity_unknown_ids", torch.as_tensor(transfer_data["unknown_ids"], dtype=torch.int64))
        self.register_buffer("_identity_anchor_ids", torch.as_tensor(transfer_data["anchor_ids"], dtype=torch.int64))
        self.register_buffer("_identity_solve_matrix", torch.as_tensor(transfer_data["solve_matrix"]))
        self.register_buffer("_identity_anchor_matrix", torch.as_tensor(transfer_data["anchor_matrix"]))
        self.register_buffer("_identity_rhs_base", torch.as_tensor(transfer_data["rhs_base"]))
        self._identity_model = {
            "mhr": self._init_mhr_identity_backend,
            "anny": self._init_anny_identity_backend,
            "smpl": self._init_linear_identity_backend,
            "smplx": self._init_linear_identity_backend,
        }[self.model_type](transfer_data)

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self.model_weights.faces

    @property
    def mean_active(self) -> Float[Tensor, "Va 3"]:
        return self.model_weights.mean_active

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return int(self.model_weights.mean_active.shape[0])

    @property
    def skin_weights(self) -> Float[Tensor, "V J"]:
        return self.model_weights.skin_weights_active[:, 1:]

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.model_weights.mean_active * 0.01

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
        identity, rest_shape_full, rest_shape_active, world_bind_pose_fit = self._prepare_identity(
            identity=identity,
            scale_params=scale_params,
            ref=pose,
        )
        return core.forward_vertices(
            data=self.model_weights,
            identity=identity,
            pose=pose,
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
            world_bind_pose_fit=world_bind_pose_fit,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
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
        identity, rest_shape_full, _rest_shape_active, world_bind_pose_fit = self._prepare_identity(
            identity=identity,
            scale_params=scale_params,
            ref=pose,
        )
        return core.forward_skeleton(
            data=self.model_weights,
            identity=identity,
            pose=pose,
            rest_shape_full=rest_shape_full,
            world_bind_pose_fit=world_bind_pose_fit,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
            match_warp=self.match_warp,
            xp=torch,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        device = self.model_weights.mean_active.device
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

    def _prepare_identity(
        self,
        *,
        identity: Float[Tensor, "B|1 I"] | None,
        scale_params: Float[Tensor, "B|1 K"] | None,
        ref: Float[Tensor, "B ..."],
    ) -> tuple[
        Float[Tensor, "B I"],
        Float[Tensor, "B V 3"],
        Float[Tensor, "B V 3"],
        Float[Tensor, "B J 4 4"],
    ]:
        return core.prepare_identity(
            data=self.model_weights,
            model_type=self.model_type,
            identity_model=self._identity_model,
            identity=identity,
            scale_params=scale_params,
            batch_size=ref.shape[0],
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
            xp=torch,
        )

    def _init_mhr_identity_backend(self, _transfer_data: dict[str, np.ndarray]) -> MHR:
        return MHR(model_path=get_identity_model_path("mhr"), simplify=1.0)

    def _init_anny_identity_backend(self, transfer_data: dict[str, np.ndarray]) -> object:
        identity_model = ANNY(
            model_path=get_identity_model_path("anny"),
            all_phenotypes=False,
            simplify=1.0,
        )
        source_vertices = torch.as_tensor(transfer_data["source_vertices"], dtype=self.model_weights.mean_full.dtype)
        rotation, translation = core.fit_rigid_transform(
            identity_model.template_vertices,
            source_vertices,
            xp=torch,
        )
        self._identity_internal_to_source_rotation = rotation
        self._identity_internal_to_source_translation = translation
        self._identity_source_to_soma_rotation = torch.as_tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=self.model_weights.mean_full.dtype,
        )
        return soma_base.AnnyIdentityData(
            template_vertices=identity_model.template_vertices,
            blendshapes=identity_model.blendshapes,
            phenotype_mask=identity_model.phenotype_mask,
            anchors=identity_model._get_anchors_dict(),
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
