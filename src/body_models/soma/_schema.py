"""Asset field names and dataclasses describing the normalized SOMA schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from jaxtyping import Float, Int

from body_models._common import kinematics, sparse
from body_models._common.skinning import CompactSkinning

SOMA_CORE_ASSET = "SOMA_neutral.npz"
SOMA_CORRECTIVES_ASSET = "correctives_model.pt"
SOMA_TEMPLATE_RIG_ASSET = "SOMA_template_rig.usda"
SOMA_PROCEDURAL_TRANSFORMS_ASSET = "SOMA_procedural_transforms.json"
SOMA_ASSETS = (SOMA_CORE_ASSET, SOMA_CORRECTIVES_ASSET)
SOMA_UPSTREAM_02_ASSETS = (SOMA_TEMPLATE_RIG_ASSET, SOMA_PROCEDURAL_TRANSFORMS_ASSET)
SOMA_LODS = ("mid", "low", "xlo")
SOMA_XLO_FIELDS = (
    "lod_mid_to_xlo",
    "triangles_xlo",
    "skinning_weights_xlo_data",
    "skinning_weights_xlo_indices",
    "skinning_weights_xlo_indptr",
    "skinning_weights_xlo_shape",
)
SOMA_RIG_FIELDS = (
    "bind_shape",
    "bind_pose_world",
    "bind_pose_local",
    "t_pose_world",
    "t_pose_local",
    "joint_parent_ids",
    "joint_names",
    "skinning_weights_data",
    "skinning_weights_indices",
    "skinning_weights_indptr",
    "skinning_weights_shape",
)
# The normalized asset schema calls the user-facing control rig "public".
SOMA_PROCEDURAL_RIG_FIELDS = (
    "public_joint_indices_full",
    "rotation_matrix",
    "translation_matrix",
    "twist_joint_indices",
    "twist_axis_ids",
    "twist_axis_signs",
    "segment_start_joint_indices",
    "segment_end_joint_indices",
    "segment_parent_joint_indices",
    "segment_reverse_indices",
    "segment_alignment_rotations",
)
SOMA_PROCEDURAL_NPZ_FIELDS = tuple(f"procedural_{name}" for name in SOMA_PROCEDURAL_RIG_FIELDS)
SOMA_NORMALIZED_NPZ_FIELDS = (
    *SOMA_RIG_FIELDS,
    *(f"public_{name}" for name in SOMA_RIG_FIELDS),
    *SOMA_PROCEDURAL_NPZ_FIELDS,
)


@dataclass(frozen=True)
class _SparseCoo:
    indices: Int[np.ndarray, "D NNZ"]
    values: Float[np.ndarray, "NNZ"]
    size: tuple[int, ...]
    is_coalesced: bool


@dataclass(frozen=True)
class SomaCorrectives:
    corrective_bindpose: Float[np.ndarray, "Jf 3 3"]
    hidden_weights: Float[np.ndarray, "input hidden"]
    basis: sparse.SparseMatrix


@dataclass(frozen=True)
class SomaKinematics:
    """Tree traversal and orientation-gather state for one SOMA rig."""

    kinematic_tree: kinematics.KinematicTree
    orientation_parent_indices: Int[np.ndarray, "Jf"]


@dataclass(frozen=True)
class SomaProceduralRig:
    control_joint_indices_full: Int[np.ndarray, "Jp"]
    rotation_matrix: Float[np.ndarray, "T Jp"]
    translation_matrix: Float[np.ndarray, "Jf Jf"]
    twist_joint_indices: Int[np.ndarray, "T"]
    twist_axis_ids: Int[np.ndarray, "T"]
    twist_axis_signs: Float[np.ndarray, "T"]
    segment_start_joint_indices: Int[np.ndarray, "S"]
    segment_end_joint_indices: Int[np.ndarray, "S"]
    segment_parent_joint_indices: Int[np.ndarray, "S"]
    segment_reverse_indices: Int[np.ndarray, "R"]
    segment_alignment_rotations: Float[np.ndarray, "S 3 3"]


@dataclass(frozen=True)
class SomaControlRig:
    joint_names_full: list[str]
    bind_pose_world: Float[np.ndarray, "Jp 4 4"]
    bind_pose_local: Float[np.ndarray, "Jp 4 4"]
    t_pose_world: Float[np.ndarray, "Jp 4 4"]
    t_pose_local: Float[np.ndarray, "Jp 4 4"]
    joint_regressor: Float[np.ndarray, "Jp Vf"]
    skin_weights_full: Float[np.ndarray, "Vf Jp"]
    skin_weights_active: Float[np.ndarray, "Va Jp"]
    kinematics: SomaKinematics
    joint_children_full: list[list[int]]
    joint_children_indices_full: Int[np.ndarray, "Jp C"]
    skinned_vertex_indices_full: list[list[int]]
    skinned_vertex_indices_full_index: Int[np.ndarray, "Jp K"]
    procedural: SomaProceduralRig


@dataclass(frozen=True)
class SomaLodMesh:
    vertex_map: Int[np.ndarray, "Va"]
    faces: Int[np.ndarray, "F 3"]
    skin_weights: Float[np.ndarray, "Va Jf"] | None = None


@dataclass(frozen=True)
class SomaAssets:
    """SOMA assets loaded from the normalized schema."""

    mean_full: Float[np.ndarray, "Vf 3"]
    mean_active: Float[np.ndarray, "Va 3"]
    shapedirs_full: Float[np.ndarray, "S Vf 3"]
    shapedirs_active: Float[np.ndarray, "S Va 3"]
    eigenvalues: Float[np.ndarray, "S"]
    bind_shape_full: Float[np.ndarray, "Vf 3"]
    bind_pose_world: Float[np.ndarray, "Jf 4 4"]
    t_pose_world: Float[np.ndarray, "Jf 4 4"]
    skin_weights_full: Float[np.ndarray, "Vf Jf"]
    skin_weights_active: Float[np.ndarray, "Va Jf"]
    compact_skinning: CompactSkinning
    faces: Int[np.ndarray, "F 3"]
    vertex_map: Int[np.ndarray, "Va"] | None
    facial_inner_vertices: Int[np.ndarray, "Va"]
    kinematics: SomaKinematics
    correctives: SomaCorrectives
    control_rig: SomaControlRig
    lods: dict[str, SomaLodMesh] | None = None


@dataclass(frozen=True)
class SomaIdentityTransfer:
    source_vertices: Float[np.ndarray, "Vs 3"]
    source_tetrahedra: Int[np.ndarray, "Fs 4"]
    face_ids: Int[np.ndarray, "Vt"]
    bary_coords: Float[np.ndarray, "Vt 4"]
    unknown_ids: Int[np.ndarray, "U"]
    anchor_ids: Int[np.ndarray, "A"]
    solve_matrix: Float[np.ndarray, "U U"]
    anchor_matrix: Float[np.ndarray, "U A"]
    rhs_base: Float[np.ndarray, "U 3"]
    internal_to_source_rotation: Float[np.ndarray, "3 3"]
    internal_to_source_translation: Float[np.ndarray, "3"]
    source_to_soma_rotation: Float[np.ndarray, "3 3"]
    source_scale: float
    output_scale: float


@dataclass(frozen=True)
class _ModelTypeSpec:
    num_shape_coeffs: int
    num_scale_coeffs: int | None = None
    default_identity_value: float = 0.0
    identity_model_kwargs: dict[str, Any] = field(default_factory=dict)
    source_scale: float = 1.0
    output_scale: float = 1.0
    asset_dir: str | None = None
    source_mesh_name: str | None = None
    target_mesh_name: str | None = None
    use_laplacian: bool = True


MODEL_TYPE_SPECS = {
    "soma": _ModelTypeSpec(num_shape_coeffs=128),
    "mhr": _ModelTypeSpec(
        num_shape_coeffs=45,
        num_scale_coeffs=68,
        source_scale=100.0,
        asset_dir="MHR",
        source_mesh_name="base_body_lod1.obj",
        target_mesh_name="SOMA_wrap_lod1.obj",
    ),
    "anny": _ModelTypeSpec(
        num_shape_coeffs=6,
        default_identity_value=0.5,
        output_scale=100.0,
        asset_dir="Anny",
        identity_model_kwargs={"all_phenotypes": False},
        source_mesh_name="base_body.obj",
        target_mesh_name="SOMA_wrap.obj",
        use_laplacian=False,
    ),
    "smpl": _ModelTypeSpec(
        num_shape_coeffs=10,
        output_scale=100.0,
        asset_dir="SMPL",
        identity_model_kwargs={"gender": "neutral"},
        source_mesh_name="base_body.obj",
        target_mesh_name="SOMA_wrap.obj",
    ),
    "smplx": _ModelTypeSpec(
        num_shape_coeffs=10,
        output_scale=100.0,
        asset_dir="SMPLX",
        identity_model_kwargs={"gender": "neutral"},
        source_mesh_name="base_body.obj",
        target_mesh_name="SOMA_wrap.obj",
    ),
}
IDENTITY_MODEL_TYPES = tuple(name for name, spec in MODEL_TYPE_SPECS.items() if spec.asset_dir is not None)
