# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0.0",
#   "scipy>=1.13.0",
#   "usd-core>=26.5",
# ]
# ///
"""Generate normalized SOMA runtime assets from upstream SOMA-X 0.2.1 assets."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, cast

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

import numpy as np  # noqa: E402
from scipy.sparse import csc_matrix  # noqa: E402

SOMA_CORE_ASSET = "SOMA_neutral.npz"
SOMA_CORRECTIVES_ASSET = "correctives_model.pt"
SOMA_TEMPLATE_RIG_ASSET = "SOMA_template_rig.usda"
SOMA_PROCEDURAL_TRANSFORMS_ASSET = "SOMA_procedural_transforms.json"
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
SOMA_PROCEDURAL_RIG_FIELDS = (
    "public_joint_indices_full",
    "rotation_matrix",
    "translation_matrix",
    "source_axis_ids",
    "source_axis_signs",
    "twist_joint_indices",
    "twist_axis_ids",
    "twist_axis_signs",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    generate_asset(args.upstream_dir, args.output_dir)


def generate_asset(upstream_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    rig_data = _load_soma_02_rig_data(upstream_dir)
    output_npz = output_dir / SOMA_CORE_ASSET
    with np.load(upstream_dir / SOMA_CORE_ASSET, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}

    for name in SOMA_RIG_FIELDS:
        arrays[name] = rig_data[name]
    public_rig = rig_data["public_rig_data"]
    for name in SOMA_RIG_FIELDS:
        arrays[f"public_{name}"] = public_rig[name]

    procedural = rig_data["procedural"]
    for name in SOMA_PROCEDURAL_RIG_FIELDS:
        arrays[f"procedural_{name}"] = procedural[name]
    arrays.update(_build_xlo_lod_arrays(upstream_dir, arrays))
    np.savez(output_npz, **arrays)

    correctives = upstream_dir / SOMA_CORRECTIVES_ASSET
    if correctives.exists():
        shutil.copy2(correctives, output_dir / SOMA_CORRECTIVES_ASSET)

    print(f"SOMA: generated normalized asset at {output_npz}")
    print(f"SOMA: to reuse it directly, run `body-models set soma {output_dir}`")
    return output_dir


def _joint_world_to_local(world: np.ndarray, parents: np.ndarray) -> np.ndarray:
    local = np.empty_like(world)
    for joint, parent in enumerate(parents.tolist()):
        local[joint] = world[joint] if parent == joint else np.linalg.inv(world[parent]) @ world[joint]
    return local


def _forward_kinematics(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    world = np.empty_like(local)
    for joint, parent in enumerate(parents.tolist()):
        world[joint] = local[joint] if parent == joint else world[parent] @ local[joint]
    return world


def _nearest_kept_parent(parent_ids: np.ndarray, old_index: int, keep_ids: set[int]) -> int:
    parent = int(parent_ids[old_index])
    while parent not in keep_ids and parent != int(parent_ids[parent]):
        parent = int(parent_ids[parent])
    return parent


def _find_soma_skin_mesh(stage: Any, lod: str = "mid") -> Any:
    from pxr import UsdGeom as _UsdGeom

    UsdGeom = cast(Any, _UsdGeom)
    mesh_name = {"mid": "c_skin_mid", "xlo": "c_skin_xlo"}[lod]
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and prim.GetPath().name == mesh_name:
            return prim
    mesh_names = sorted(prim.GetPath().name for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh))
    names = ", ".join(mesh_names)
    raise ValueError(
        f"Could not find SOMA {lod} skin mesh {mesh_name!r} in {SOMA_TEMPLATE_RIG_ASSET}. Available meshes: {names}"
    )


def _load_soma_02_rig_from_usd(asset_dir: Path) -> dict[str, Any]:
    from pxr import Usd as _Usd
    from pxr import UsdGeom as _UsdGeom
    from pxr import UsdSkel as _UsdSkel

    Usd = cast(Any, _Usd)
    UsdGeom = cast(Any, _UsdGeom)
    UsdSkel = cast(Any, _UsdSkel)

    usd_path = asset_dir / SOMA_TEMPLATE_RIG_ASSET
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Failed to open SOMA template rig: {usd_path}")

    skel_prim = next((prim for prim in stage.Traverse() if prim.IsA(UsdSkel.Skeleton)), None)
    if skel_prim is None:
        raise RuntimeError(f"No UsdSkelSkeleton prim found in SOMA template rig: {usd_path}")
    skeleton = UsdSkel.Skeleton(skel_prim)

    joint_paths = list(skeleton.GetJointsAttr().Get() or [])
    bind_xforms = skeleton.GetBindTransformsAttr().Get()
    rest_xforms = skeleton.GetRestTransformsAttr().Get()
    if not joint_paths or bind_xforms is None or rest_xforms is None:
        raise RuntimeError(f"SOMA template rig is missing skeleton transforms: {usd_path}")

    parent_lookup = {path: idx for idx, path in enumerate(joint_paths)}
    parent_ids = np.zeros((len(joint_paths),), dtype=np.int32)
    for index, path in enumerate(joint_paths):
        if "/" in path:
            parent_ids[index] = parent_lookup[path.rsplit("/", 1)[0]]

    bind_pose_world = np.asarray(bind_xforms, dtype=np.float32).reshape(len(joint_paths), 4, 4).swapaxes(-2, -1)
    t_pose_local = np.asarray(rest_xforms, dtype=np.float32).reshape(len(joint_paths), 4, 4).swapaxes(-2, -1)
    bind_pose_local = _joint_world_to_local(bind_pose_world, parent_ids)
    t_pose_world = _forward_kinematics(t_pose_local, parent_ids)

    skin_prim = _find_soma_skin_mesh(stage)
    bind_shape = np.asarray(UsdGeom.Mesh(skin_prim).GetPointsAttr().Get(), dtype=np.float32)
    binding = UsdSkel.BindingAPI(skin_prim)
    joint_indices = binding.GetJointIndicesPrimvar()
    joint_weights = binding.GetJointWeightsPrimvar()
    if not joint_indices or not joint_weights:
        raise RuntimeError(f"SOMA skin mesh has no skinning primvars: {skin_prim.GetPath()}")

    num_vertices = bind_shape.shape[0]
    num_weights = joint_indices.GetElementSize()
    indices = np.asarray(joint_indices.Get(), dtype=np.int32).reshape(num_vertices, num_weights)
    weights = np.asarray(joint_weights.Get(), dtype=np.float32).reshape(num_vertices, num_weights)
    joint_path_to_index = {path: idx for idx, path in enumerate(joint_paths)}
    binding_to_skeleton = np.asarray(
        [joint_path_to_index[str(path)] for path in binding.GetJointsAttr().Get()], dtype=np.int32
    )

    vertex_indices = np.repeat(np.arange(num_vertices, dtype=np.int32), num_weights)
    skeleton_indices = binding_to_skeleton[indices.ravel()]
    values = weights.ravel()
    dense_weights = np.zeros((num_vertices, len(joint_paths)), dtype=np.float32)
    np.add.at(dense_weights, (vertex_indices[values > 0], skeleton_indices[values > 0]), values[values > 0])
    sparse_weights = csc_matrix(dense_weights)

    return {
        "joint_names": np.asarray([path.split("/")[-1] for path in joint_paths]),
        "joint_parent_ids": parent_ids,
        "bind_pose_world": bind_pose_world.astype(np.float32),
        "bind_pose_local": bind_pose_local.astype(np.float32),
        "t_pose_world": t_pose_world.astype(np.float32),
        "t_pose_local": t_pose_local.astype(np.float32),
        "bind_shape": bind_shape.astype(np.float32),
        "skinning_weights_data": sparse_weights.data.astype(np.float32),
        "skinning_weights_indices": sparse_weights.indices.astype(np.int32),
        "skinning_weights_indptr": sparse_weights.indptr.astype(np.int32),
        "skinning_weights_shape": np.asarray(sparse_weights.shape, dtype=np.int32),
    }


def _load_lod_skin_from_usd(asset_dir: Path, lod: str) -> dict[str, Any]:
    from pxr import Usd as _Usd
    from pxr import UsdGeom as _UsdGeom
    from pxr import UsdSkel as _UsdSkel

    Usd = cast(Any, _Usd)
    UsdGeom = cast(Any, _UsdGeom)
    UsdSkel = cast(Any, _UsdSkel)

    usd_path = asset_dir / SOMA_TEMPLATE_RIG_ASSET
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Failed to open SOMA template rig: {usd_path}")
    skel_prim = next((prim for prim in stage.Traverse() if prim.IsA(UsdSkel.Skeleton)), None)
    if skel_prim is None:
        raise RuntimeError(f"No UsdSkelSkeleton prim found in SOMA template rig: {usd_path}")
    joint_paths = list(UsdSkel.Skeleton(skel_prim).GetJointsAttr().Get() or [])
    skin_prim = _find_soma_skin_mesh(stage, lod)
    mesh = UsdGeom.Mesh(skin_prim)
    bind_shape = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    face_counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    face_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    triangles = _fan_triangulate(face_indices, face_counts)

    binding = UsdSkel.BindingAPI(skin_prim)
    joint_indices = binding.GetJointIndicesPrimvar()
    joint_weights = binding.GetJointWeightsPrimvar()
    if not joint_indices or not joint_weights:
        raise RuntimeError(f"SOMA {lod} skin mesh has no skinning primvars: {skin_prim.GetPath()}")

    num_vertices = bind_shape.shape[0]
    num_weights = joint_indices.GetElementSize()
    indices = np.asarray(joint_indices.Get(), dtype=np.int32).reshape(num_vertices, num_weights)
    weights = np.asarray(joint_weights.Get(), dtype=np.float32).reshape(num_vertices, num_weights)
    joint_path_to_index = {path: idx for idx, path in enumerate(joint_paths)}
    binding_to_skeleton = np.asarray(
        [joint_path_to_index[str(path)] for path in binding.GetJointsAttr().Get()], dtype=np.int32
    )

    vertex_indices = np.repeat(np.arange(num_vertices, dtype=np.int32), num_weights)
    skeleton_indices = binding_to_skeleton[indices.ravel()]
    values = weights.ravel()
    dense_weights = np.zeros((num_vertices, len(joint_paths)), dtype=np.float32)
    np.add.at(dense_weights, (vertex_indices[values > 0], skeleton_indices[values > 0]), values[values > 0])
    return {"bind_shape": bind_shape, "triangles": triangles, "skin_weights": dense_weights}


def _fan_triangulate(face_indices: np.ndarray, face_counts: np.ndarray) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for count in face_counts.tolist():
        face = face_indices[offset : offset + count]
        offset += count
        for index in range(1, count - 1):
            triangles.append((int(face[0]), int(face[index]), int(face[index + 1])))
    return np.asarray(triangles, dtype=np.int64)


def _build_xlo_lod_arrays(
    upstream_dir: Path,
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    xlo = _load_lod_skin_from_usd(upstream_dir, "xlo")
    mid_bind_shape = np.asarray(arrays["bind_shape"], dtype=np.float32)
    nearest_mid = _nearest_vertex_ids(mid_bind_shape, xlo["bind_shape"])
    xlo_skin_weights = csc_matrix(xlo["skin_weights"])

    return {
        "lod_mid_to_xlo": nearest_mid,
        "triangles_xlo": xlo["triangles"].astype(np.int64),
        "skinning_weights_xlo_data": xlo_skin_weights.data.astype(np.float32),
        "skinning_weights_xlo_indices": xlo_skin_weights.indices.astype(np.int32),
        "skinning_weights_xlo_indptr": xlo_skin_weights.indptr.astype(np.int32),
        "skinning_weights_xlo_shape": np.asarray(xlo_skin_weights.shape, dtype=np.int32),
    }


def _nearest_vertex_ids(source: np.ndarray, target: np.ndarray, chunk_size: int = 128) -> np.ndarray:
    nearest = np.empty((target.shape[0],), dtype=np.int64)
    for start in range(0, target.shape[0], chunk_size):
        stop = min(start + chunk_size, target.shape[0])
        distances = np.sum((target[start:stop, None] - source[None]) ** 2, axis=-1)
        nearest[start:stop] = np.argmin(distances, axis=1)
    return nearest


def _axis_id(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis.lower()]


def _load_procedural_data(asset_dir: Path, joint_names: list[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    with (asset_dir / SOMA_PROCEDURAL_TRANSFORMS_ASSET).open() as file:
        data = json.load(file)

    public_names = data["public_rig_derivation"]["main_joint_names"]
    joint_index = {name: index for index, name in enumerate(joint_names)}
    public_index = {name: index for index, name in enumerate(public_names)}
    source_axis_ids = np.zeros((len(public_names),), dtype=np.int64)
    source_axis_signs = np.ones((len(public_names),), dtype=np.float32)
    twist_names: list[str] = []
    twist_axis_ids: list[int] = []
    twist_axis_signs: list[float] = []
    for segment in data["segments"]:
        axis = _axis_id(segment["source_axis"])
        sign = float(segment["source_sign"])
        for source_name in (segment["start_joint"], segment["end_joint"]):
            if source_name in public_index:
                source_axis_ids[public_index[source_name]] = axis
                source_axis_signs[public_index[source_name]] = sign
        for twist_name in segment["twist_joints"]:
            twist_names.append(twist_name)
            twist_axis_ids.append(axis)
            twist_axis_signs.append(sign)

    twist_index = {name: index for index, name in enumerate(twist_names)}
    rotation_matrix = np.zeros((len(twist_names), len(public_names)), dtype=np.float32)
    for entry in data["parameter_matrices"]["rotation"]["entries"]:
        rotation_matrix[twist_index[entry["row"]], public_index[entry["column"]]] = float(entry["value"])

    translation_matrix = np.eye(len(joint_names), dtype=np.float32)
    cleared_rows: set[int] = set()
    for entry in data["parameter_matrices"]["translation"]["entries"]:
        row = joint_index[entry["row"]]
        if row not in cleared_rows:
            translation_matrix[row] = 0.0
            cleared_rows.add(row)
        translation_matrix[row, joint_index[entry["column"]]] = float(entry["value"])

    return public_names, {
        "public_joint_indices_full": np.asarray([joint_index[name] for name in public_names], dtype=np.int64),
        "rotation_matrix": rotation_matrix,
        "translation_matrix": translation_matrix,
        "source_axis_ids": source_axis_ids,
        "source_axis_signs": source_axis_signs,
        "twist_joint_indices": np.asarray([joint_index[name] for name in twist_names], dtype=np.int64),
        "twist_axis_ids": np.asarray(twist_axis_ids, dtype=np.int64),
        "twist_axis_signs": np.asarray(twist_axis_signs, dtype=np.float32),
    }


def _derive_public_rig(rig_data: dict[str, Any], public_joint_names: list[str]) -> dict[str, Any]:
    joint_names = [str(name) for name in rig_data["joint_names"]]
    keep_ids = np.asarray([joint_names.index(name) for name in public_joint_names], dtype=np.int64)
    keep_id_set = {int(index) for index in keep_ids}
    parent_ids = np.asarray(rig_data["joint_parent_ids"], dtype=np.int64)
    old_to_new = {int(old_index): new_index for new_index, old_index in enumerate(keep_ids)}

    new_parent_ids = np.zeros((len(keep_ids),), dtype=np.int32)
    for new_index, old_index_np in enumerate(keep_ids):
        old_index = int(old_index_np)
        parent = int(parent_ids[old_index])
        if parent == old_index:
            new_parent_ids[new_index] = new_index
        else:
            new_parent_ids[new_index] = old_to_new[_nearest_kept_parent(parent_ids, old_index, keep_id_set)]

    dense_weights = csc_matrix(
        (
            rig_data["skinning_weights_data"],
            rig_data["skinning_weights_indices"],
            rig_data["skinning_weights_indptr"],
        ),
        shape=tuple(int(x) for x in rig_data["skinning_weights_shape"]),
    ).toarray()
    for removed_index in sorted(set(range(len(joint_names))) - keep_id_set):
        dense_weights[:, _nearest_kept_parent(parent_ids, removed_index, keep_id_set)] += dense_weights[
            :, removed_index
        ]
    sparse_weights = csc_matrix(dense_weights[:, keep_ids])
    bind_pose_world = np.asarray(rig_data["bind_pose_world"], dtype=np.float32)[keep_ids]
    t_pose_world = np.asarray(rig_data["t_pose_world"], dtype=np.float32)[keep_ids]

    out = dict(rig_data)
    out.update(
        joint_names=np.asarray(public_joint_names),
        joint_parent_ids=new_parent_ids,
        bind_pose_world=bind_pose_world,
        bind_pose_local=_joint_world_to_local(bind_pose_world, new_parent_ids).astype(np.float32),
        t_pose_world=t_pose_world,
        t_pose_local=_joint_world_to_local(t_pose_world, new_parent_ids).astype(np.float32),
        skinning_weights_data=sparse_weights.data.astype(np.float32),
        skinning_weights_indices=sparse_weights.indices.astype(np.int32),
        skinning_weights_indptr=sparse_weights.indptr.astype(np.int32),
        skinning_weights_shape=np.asarray(sparse_weights.shape, dtype=np.int32),
    )
    return out


def _load_soma_02_rig_data(asset_dir: Path) -> dict[str, Any]:
    expanded_rig = _load_soma_02_rig_from_usd(asset_dir)
    public_joint_names, procedural = _load_procedural_data(
        asset_dir, [str(name) for name in expanded_rig["joint_names"]]
    )
    expanded_rig["procedural"] = procedural
    expanded_rig["public_rig_data"] = _derive_public_rig(expanded_rig, public_joint_names)
    return expanded_rig


if __name__ == "__main__":
    main()
