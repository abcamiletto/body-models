"""I/O utilities for GNM Head model loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

from body_models import _config as config
from body_models._cache import download_hf_archive, get_cache_dir
from body_models._common import kinematics, simplify_mesh
from body_models._common.kinematics import compute_sparse_skin_weights
from body_models._common.skinning import CompactSkinning

PathLike = Path | str
MODEL_FILENAME = "gnm_head.npz"


@dataclass(frozen=True)
class GnmAssets:
    v_template: Float[np.ndarray, "V 3"]
    shapedirs: Float[np.ndarray, "V 3 253"]
    exprdirs: Float[np.ndarray, "V 3 383"]
    j_template: Float[np.ndarray, "4 3"]
    j_shapedirs: Float[np.ndarray, "4 3 253"]
    lbs_weights: Float[np.ndarray, "V 4"]
    compact_skinning: CompactSkinning
    posedirs: Float[np.ndarray, "36 V*3"]
    faces: Int[np.ndarray, "F 3"]
    kinematic_tree: kinematics.KinematicTree
    joint_names: list[str]
    identity_names: list[str]
    expression_names: list[str]


def validate_path(model_path: PathLike) -> Path:
    """Validate and return a GNM model file."""
    model_path = Path(model_path)
    if model_path.is_dir():
        model_path = model_path / MODEL_FILENAME
    if not model_path.is_file():
        raise FileNotFoundError(f"GNM model file not found: {model_path}")
    if model_path.suffix != ".npz":
        raise ValueError(f"Expected a GNM .npz file, got: {model_path}")
    return model_path


def get_model_path(model_path: PathLike | None = None) -> Path:
    """Resolve the GNM model path, downloading it when needed."""
    if model_path is None:
        model_path = config.get_model_path("gnm")
    if model_path is not None:
        return validate_path(model_path)

    cache_path = get_cache_dir() / "gnm" / MODEL_FILENAME
    if cache_path.is_file():
        return cache_path
    return download_model()


def download_model(output_dir: PathLike | None = None) -> Path:
    """Download the GNM Head v3.0 model assets."""
    output_dir = Path(output_dir) if output_dir is not None else get_cache_dir() / "gnm"
    model_path = output_dir / MODEL_FILENAME
    if model_path.is_file():
        return validate_path(model_path)
    print(f"Downloading GNM Head model to {output_dir}...")
    download_hf_archive("gnm/assets.zip", output_dir)
    print("Done")
    return validate_path(model_path)


def load_model_data(model_path: PathLike, *, simplify: float = 1.0) -> GnmAssets:
    """Load GNM Head v3.0 data into the shared model representation."""
    if simplify < 1.0:
        raise ValueError("simplify must be >= 1.0")

    with np.load(validate_path(model_path), allow_pickle=False) as data:
        version = str(data["version"])
        variant = str(data["variant"])
        if version != "3.0" or variant != "head":
            raise ValueError(f"Expected GNM Head v3.0, got variant={variant!r}, version={version!r}")

        v_template = np.array(data["template_vertex_positions"], copy=True)
        shapedirs = np.moveaxis(data["vertex_identity_basis"], 0, -1).copy()
        exprdirs = np.moveaxis(data["expression_basis"], 0, -1).copy()
        lbs_weights = np.asarray(data["skinning_weights"]).T.copy()
        posedirs = np.array(data["pose_correctives_regressor"], copy=True)
        faces = np.asarray(data["triangles"], dtype=np.int64).copy()
        j_template = np.array(data["template_joint_positions"], copy=True)
        j_shapedirs = np.moveaxis(data["joint_identity_basis"], 0, -1).copy()
        parents = np.asarray(data["joint_parent_indices"], dtype=np.int64)
        joint_names = [str(name) for name in data["joint_names"]]
        identity_names = [str(name) for name in data["identity_names"]]
        expression_names = [str(name) for name in data["expression_names"]]

    if simplify > 1.0:
        target_faces = int(len(faces) / simplify)
        v_template, faces, vertex_map = simplify_mesh(v_template, faces, target_faces)
        shapedirs = shapedirs[vertex_map]
        exprdirs = exprdirs[vertex_map]
        lbs_weights = lbs_weights[vertex_map]
        posedirs = posedirs.reshape(36, -1, 3)[:, vertex_map].reshape(36, -1)

    joint_indices, joint_weights = compute_sparse_skin_weights(lbs_weights)
    return GnmAssets(
        v_template=v_template,
        shapedirs=shapedirs,
        exprdirs=exprdirs,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        lbs_weights=lbs_weights,
        compact_skinning=CompactSkinning(joint_indices, joint_weights),
        posedirs=posedirs,
        faces=faces,
        kinematic_tree=kinematics.KinematicTree.from_parents(parents),
        joint_names=joint_names,
        identity_names=identity_names,
        expression_names=expression_names,
    )


__all__ = ["GnmAssets", "download_model", "get_model_path", "load_model_data", "validate_path"]
