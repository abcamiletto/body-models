import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import config
from body_models.common import simplify_mesh

PathLike = Path | str
Array = Any

__all__ = ["load_model_data"]


@dataclass(frozen=True)
class SkelWeights:
    v_template: Float[Array, "V 3"]
    faces: Int[Array, "F 3"]
    shapedirs: Float[Array, "V 3 B"]
    posedirs: Float[Array, "P V*3"]
    skin_weights: Float[Array, "V 24"]
    j_template: Float[Array, "24 3"]
    j_shapedirs: Float[Array, "24 3 B"]
    skel_v_template: Float[Array, "Vs 3"]
    skel_faces: Int[Array, "Fs 3"]
    skel_weights: Float[Array, "Vs 24"]
    skel_weights_rigid: Float[Array, "Vs 24"]
    apose_R: Float[Array, "24 3 3"]
    apose_t: Float[Array, "24 3"]
    per_joint_rot: Float[Array, "24 3 3"]
    parent: Int[Array, "23"]
    parents: list[int]
    child: Int[Array, "24"]
    fixed_orientation_joints: Int[Array, "6"]
    non_leaf_joints: Int[Array, "N"]
    non_leaf_children: Int[Array, "N"]
    all_axes: Float[Array, "47 3"]
    rotation_indices: Int[Array, "24 3"]
    scapula_r_axes: Float[Array, "3 3"]
    scapula_l_axes: Float[Array, "3 3"]
    spine_axes: Float[Array, "3 3"]
    num_joints_smpl: int
    joint_names: list[str]


def validate_path(model_path: PathLike) -> Path:
    model_path = Path(model_path)
    if model_path.is_dir():
        raise ValueError(f"Expected a SKEL model file, got directory: {model_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"SKEL model file not found: {model_path}")
    return model_path


def get_model_path(model_path: PathLike | None, gender: Literal["male", "female"] | None) -> Path:
    if gender is None:
        raise ValueError("gender must be 'male' or 'female'.")

    if model_path is None:
        model_path = config.get_model_path(f"skel-{gender.lower()}")

    if model_path is None:
        raise FileNotFoundError(
            "SKEL model not found. Download from https://skel.is.tue.mpg.de/ "
            f"and run: body-models set skel-{gender.lower()} /path/to/model.pkl"
        )

    return validate_path(model_path)


def load_model_data(model_path: Path, simplify: float = 1.0) -> SkelWeights:
    assert simplify >= 1.0

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    skin_template = np.asarray(data["skin_template_v"], dtype=np.float32)
    faces = np.asarray(data["skin_template_f"], dtype=np.int32)
    skin_shapedirs = np.asarray(data["shapedirs"][:, :, :10], dtype=np.float32)
    posedirs_full = np.asarray(data["posedirs"], dtype=np.float32)
    skin_weights = _sparse_to_dense(data["skin_weights"])
    j_regressor = _sparse_to_dense(data["J_regressor_osim"])
    j_template = j_regressor @ skin_template
    j_shapedirs = np.einsum("jv,vdb->jdb", j_regressor, skin_shapedirs)

    if simplify > 1.0:
        target_faces = int(len(faces) / simplify)
        v_template, faces, vertex_map = simplify_mesh(skin_template, faces, target_faces)
        shapedirs = skin_shapedirs[vertex_map]
        posedirs = posedirs_full[vertex_map]
        skin_weights = skin_weights[vertex_map]
    else:
        v_template = skin_template
        shapedirs = skin_shapedirs
        posedirs = posedirs_full

    kintree = np.asarray(data["osim_kintree_table"], dtype=np.int64)
    id_to_col = {kintree[1, i]: i for i in range(kintree.shape[1])}
    parent_list = [id_to_col[kintree[0, i]] for i in range(1, kintree.shape[1])]
    child = _compute_child(kintree)
    non_leaf = [i for i, child_index in enumerate(child) if child_index != 0]

    return SkelWeights(
        v_template=v_template,
        faces=faces,
        shapedirs=shapedirs,
        posedirs=posedirs.reshape(-1, posedirs.shape[-1]).T,
        skin_weights=skin_weights,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        skel_v_template=np.asarray(data["skel_template_v"], dtype=np.float32),
        skel_faces=np.asarray(data["skel_template_f"], dtype=np.int32),
        skel_weights=_sparse_to_dense(data["skel_weights"]),
        skel_weights_rigid=_sparse_to_dense(data["skel_weights_rigid"]),
        apose_R=np.asarray(data["apose_rel_transfo"], dtype=np.float32)[:, :3, :3],
        apose_t=np.asarray(data["apose_rel_transfo"], dtype=np.float32)[:, :3, 3],
        per_joint_rot=np.asarray(data["per_joint_rot"], dtype=np.float32),
        parent=np.asarray(parent_list, dtype=np.int64),
        parents=[-1, *parent_list],
        child=child,
        fixed_orientation_joints=np.asarray([0, 5, 10, 13, 18, 23], dtype=np.int64),
        non_leaf_joints=np.asarray(non_leaf, dtype=np.int64),
        non_leaf_children=np.asarray([child[i] for i in non_leaf], dtype=np.int64),
        all_axes=_compute_joint_axes(),
        rotation_indices=_compute_rotation_indices(),
        scapula_r_axes=np.asarray([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=np.float32),
        scapula_l_axes=np.asarray([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32),
        spine_axes=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32),
        num_joints_smpl=data["J_regressor"].shape[0],
        joint_names=list(data["bone_names"]),
    )


def _sparse_to_dense(matrix) -> np.ndarray:
    return matrix.toarray().astype(np.float32)


def _compute_child(kintree: Int[np.ndarray, "2 24"]) -> Int[np.ndarray, "24"]:
    child = []
    for joint in range(24):
        children = np.where(kintree[0] == joint)[0]
        child.append(kintree[1, children[0]] if len(children) > 0 else 0)
    return np.asarray(child, dtype=np.int64)


def _compute_joint_axes() -> Float[np.ndarray, "47 3"]:
    joint_axes = [
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        [[0, 0, -1]],
        [_pin_axis_from_euler([0.175895, -0.105208, 0.0186622])],
        [_pin_axis_from_euler([-1.76818999, 0.906223, 1.8196])],
        [_pin_axis_from_euler([-3.14158999, 0.619901, 0])],
        [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
        [[0, 0, -1]],
        [_pin_axis_from_euler([0.175895, -0.105208, 0.0186622])],
        [_pin_axis_from_euler([1.76818999, -0.906223, 1.8196])],
        [_pin_axis_from_euler([-3.14158999, -0.619901, 0])],
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0.0494, 0.0366, 0.99810825]],
        [[-0.01716099, 0.99266564, -0.11966796]],
        [[1, 0, 0], [0, 0, -1]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[-0.0494, -0.0366, 0.99810825]],
        [[0.01716099, -0.99266564, -0.11966796]],
        [[-1, 0, 0], [0, 0, -1]],
    ]
    axes = [_normalize_axis(axis) for joint in joint_axes for axis in joint]
    axes.append([0.0, 0.0, 0.0])
    return np.asarray(axes, dtype=np.float32)


def _compute_rotation_indices() -> Int[np.ndarray, "24 3"]:
    indices = []
    dof_idx = 0
    for dofs in [3, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1, 1, 2, 3, 3, 1, 1, 2]:
        identity_idx = 46
        if dofs == 1:
            indices.append([dof_idx, identity_idx, identity_idx])
        elif dofs == 2:
            indices.append([dof_idx, dof_idx + 1, identity_idx])
        else:
            indices.append([dof_idx, dof_idx + 1, dof_idx + 2])
        dof_idx += dofs
    return np.asarray(indices, dtype=np.int64)


def _pin_axis_from_euler(euler_xyz: list[float]) -> list[float]:
    euler = np.asarray(euler_xyz, dtype=np.float32)
    rotation = SO3.conversions.from_euler_to_rotmat(euler, convention="XYZ")
    axis = rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return axis.tolist()


def _normalize_axis(axis: list[float]) -> list[float]:
    array = np.asarray(axis, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm > 1e-6 and abs(norm - 1.0) > 1e-6:
        array = array / norm
    return array.tolist()
