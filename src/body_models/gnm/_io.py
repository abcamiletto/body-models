"""I/O utilities for GNM Head model loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

from body_models import _config as config
from body_models import _download
from body_models._common import kinematics, simplify_mesh, skinning

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
    compact_skinning: skinning.CompactSkinning
    posedirs: Float[np.ndarray, "36 V*3"]
    faces: Int[np.ndarray, "F 3"]
    kinematic_tree: kinematics.KinematicTree
    joint_names: list[str]
    identity_names: list[str]
    expression_names: list[str]


def validate_path(model_path: PathLike) -> Path:
    return config.validate_file(model_path, "GNM", (".npz",))


def download_model(output_dir: PathLike | None = None) -> Path:
    """Download the public GNM Head v3.0 assets unless they are already present."""
    return _download.download_public_assets(
        "gnm", output_dir, lambda directory: validate_path(directory / MODEL_FILENAME)
    )


def load_model_data(model_path: PathLike, *, simplify: float = 1.0) -> GnmAssets:
    """Load GNM Head v3.0 data into the shared model representation."""
    if simplify < 1.0:
        raise ValueError("simplify must be >= 1.0")

    with np.load(model_path, allow_pickle=False) as data:
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
        v_template, faces, vertex_map = simplify_mesh.simplify_mesh(v_template, faces, target_faces)
        shapedirs = shapedirs[vertex_map]
        exprdirs = exprdirs[vertex_map]
        lbs_weights = lbs_weights[vertex_map]
        posedirs = posedirs.reshape(36, -1, 3)[:, vertex_map].reshape(36, -1)

    joint_indices, joint_weights = kinematics.compute_sparse_skin_weights(lbs_weights)
    return GnmAssets(
        v_template=v_template,
        shapedirs=shapedirs,
        exprdirs=exprdirs,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        lbs_weights=lbs_weights,
        compact_skinning=skinning.CompactSkinning(joint_indices, joint_weights),
        posedirs=posedirs,
        faces=faces,
        kinematic_tree=kinematics.KinematicTree.from_parents(parents),
        joint_names=joint_names,
        identity_names=identity_names,
        expression_names=expression_names,
    )


__all__ = ["GnmAssets", "download_model", "load_model_data", "validate_path"]
