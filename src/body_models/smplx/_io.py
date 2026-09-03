"""I/O utilities for SMPL-X model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import trimesh
from jaxtyping import Float, Int

from body_models import _config as config
from body_models._common import compute_sparse_skin_weights, kinematics, simplify_mesh
from body_models._common.chumpy_fix import load_model_dict
from body_models._common.skinning import CompactSkinning

PathLike = Path | str
Array = Any

__all__ = ["load_model_data"]


@dataclass(frozen=True)
class SmplxAssets:
    v_template: Float[Array, "V 3"]
    faces: Int[Array, "F 3"]
    lbs_weights: Float[Array, "V 55"]
    compact_skinning: CompactSkinning
    shapedirs: Float[Array, "V 3 S"]
    exprdirs: Float[Array, "V 3 E"]
    posedirs: Float[Array, "P V*3"]
    j_template: Float[Array, "55 3"]
    j_shapedirs: Float[Array, "55 3 S"]
    j_exprdirs: Float[Array, "55 3 E"]
    hand_mean: Float[Array, "2 45"]
    kinematic_tree: kinematics.KinematicTree
    joint_names: list[str]


def validate_path(model_path: PathLike) -> Path:
    model_path = Path(model_path)
    if model_path.is_dir():
        raise ValueError(f"Expected an SMPLX model file, got directory: {model_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPLX model file not found: {model_path}")
    if model_path.suffix not in {".npz", ".pkl", ".obj"}:
        raise ValueError(f"Expected an SMPLX .npz, .pkl or .obj file, got: {model_path}")
    return model_path


def load_toeless_template() -> Float[np.ndarray, "V 3"]:
    """Load the registered BEDLAM2 toeless vertex template."""
    template_path = config.get_model_path("template-smplx-neutral-toeless")
    if template_path is None:
        raise FileNotFoundError(
            "SMPL-X toeless template not found. Configure it with "
            "`body-models set template-smplx-neutral-toeless /path/to/smplx_neutral-lh_vtemplate_toeless.obj`."
        )
    mesh = trimesh.load_mesh(template_path, process=False, maintain_order=True)
    return np.asarray(mesh.vertices, dtype=np.float32)


def get_model_path(model_path: PathLike | None, gender: Literal["neutral", "male", "female"] | None) -> Path:
    if model_path is not None:
        if gender is not None:
            raise ValueError("gender is only supported when model_path is not provided.")
        return validate_path(model_path)

    if gender is None:
        raise ValueError("Either model_path or gender must be provided.")

    config_key = f"smplx-{gender}"
    resolved_path = config.get_model_path(config_key)

    if resolved_path is None:
        raise FileNotFoundError(
            "SMPL-X model not found. Run `body-models download smplx` or configure an existing file with "
            f"`body-models set smplx-{gender} /path/to/SMPLX_{gender.upper()}.npz`."
        )

    return validate_path(resolved_path)


def load_model_data(
    path: Path,
    flat_hand_mean: bool = False,
    simplify: float = 1.0,
    v_template: Float[np.ndarray, "V 3"] | None = None,
) -> SmplxAssets:
    """Load SMPL-X model data from .pkl or .npz file, optionally replacing the vertex template."""
    if simplify < 1.0:
        raise ValueError("simplify must be >= 1.0")
    data = load_model_dict(path)

    template = data["v_template"] if v_template is None else v_template
    model_template = np.asarray(template, dtype=np.float32)
    faces = np.asarray(data["f"], dtype=np.int32)
    lbs_weights = np.asarray(data["weights"], dtype=np.float32)
    model_dirs = np.asarray(data["shapedirs"], dtype=np.float32)
    posedirs = np.asarray(data["posedirs"], dtype=np.float32)
    joint_regressor = np.asarray(data["J_regressor"], dtype=np.float32)
    parents = np.asarray(data["kintree_table"][0], dtype=np.int64)
    parents[0] = -1

    vertices = model_template
    shapedirs = model_dirs
    if simplify > 1.0:
        target_faces = int(len(faces) / simplify)
        vertices, faces, vertex_map = simplify_mesh(model_template, faces, target_faces)
        lbs_weights = lbs_weights[vertex_map]
        shapedirs = model_dirs[vertex_map]
        posedirs = posedirs[vertex_map]

    hand_mean = np.stack(
        [
            np.asarray(data["hands_meanl"], dtype=np.float32),
            np.asarray(data["hands_meanr"], dtype=np.float32),
        ]
    )
    if flat_hand_mean:
        hand_mean = np.zeros_like(hand_mean)

    lbs_joint_indices, lbs_joint_weights = compute_sparse_skin_weights(lbs_weights)

    return SmplxAssets(
        v_template=vertices,
        faces=faces,
        lbs_weights=lbs_weights,
        compact_skinning=CompactSkinning(lbs_joint_indices, lbs_joint_weights),
        shapedirs=shapedirs[:, :, :300],
        exprdirs=shapedirs[:, :, 300:400],
        posedirs=posedirs.reshape(-1, posedirs.shape[-1]).T,
        j_template=joint_regressor @ model_template,
        j_shapedirs=np.einsum("jv,vds->jds", joint_regressor, model_dirs[:, :, :300]),
        j_exprdirs=np.einsum("jv,vde->jde", joint_regressor, model_dirs[:, :, 300:400]),
        hand_mean=hand_mean,
        kinematic_tree=kinematics.KinematicTree.from_parents(parents),
        joint_names=get_joint_names(data),
    )


def get_joint_names(model_data: dict) -> list[str]:
    """Extract ordered SMPL-X joint names from model data."""
    joint2num = model_data["joint2num"]
    if isinstance(joint2num, np.ndarray):
        joint2num = joint2num.item()
    return [name for name, _ in sorted(joint2num.items(), key=lambda item: int(item[1]))]
