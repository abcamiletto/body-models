# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "body-models",
#   "usd-core>=26.5",
# ]
# [tool.uv.sources]
# body-models = { path = "../.." }
# ///
"""Generate the body-models SOMA assets from a SOMA-X Hugging Face snapshot.

Usage:

    hf download nvidia/SOMA-X --revision v0.3.6 --local-dir /path/to/upstream
    uv run scripts/preprocess/soma.py /path/to/upstream /path/to/output

The output folder is what ``soma/assets.zip`` holds on Hugging Face.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from jaxtyping import Float, Int
from pxr import Usd, UsdGeom, UsdSkel
from scipy.sparse import csc_matrix

from body_models.soma._schema import MODEL_TYPE_SPECS, SOMA_CORE_ASSET, SOMA_CORRECTIVES_ASSET, SOMA_X_VERSION

SOMA_X_MANIFEST = "SOMA-X-HF-MANIFEST.json"
SOMA_TEMPLATE_RIG_ASSET = "SOMA_template_rig.usda"
SOMA_PROCEDURAL_TRANSFORMS_ASSET = "SOMA_procedural_transforms.json"


def main(
    upstream_dir: Annotated[Path, typer.Argument()],
    output_dir: Annotated[Path, typer.Argument()],
) -> None:
    release = json.loads((upstream_dir / SOMA_X_MANIFEST).read_text())["release_tag"]
    if release != f"v{SOMA_X_VERSION}":
        raise ValueError(f"{upstream_dir} holds SOMA-X {release}; body-models needs SOMA-X v{SOMA_X_VERSION}.")

    stage = Usd.Stage.Open(str(upstream_dir / SOMA_TEMPLATE_RIG_ASSET))
    rig = _load_rig(stage)
    procedural = json.loads((upstream_dir / SOMA_PROCEDURAL_TRANSFORMS_ASSET).read_text())
    public_names = procedural["public_rig_derivation"]["main_joint_names"]
    public_rig = _public_rig(rig, public_names)
    xlo_shape, xlo_triangles, xlo_weights = _load_skin(stage, rig["joint_paths"], "c_skin_xlo")

    with np.load(upstream_dir / SOMA_CORE_ASSET, allow_pickle=False) as data:
        # Shape space, meshes and LOD maps; the reference T-poses of earlier releases are dropped.
        arrays = {name: data[name] for name in data.files if not name.startswith(("t_pose_world__", "reference_pose_"))}
    arrays["soma_x_version"] = np.asarray(SOMA_X_VERSION)
    arrays |= _rig_arrays(rig, prefix="")
    arrays |= _rig_arrays(public_rig, prefix="public_")
    arrays |= {f"procedural_{name}": value for name, value in _procedural_arrays(procedural, rig, public_rig).items()}
    arrays["lod_mid_to_xlo"] = _nearest_vertex_ids(rig["bind_shape"], xlo_shape)
    arrays["triangles_xlo"] = xlo_triangles
    arrays |= _sparse_arrays(xlo_weights, "skinning_weights_xlo")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / SOMA_CORE_ASSET, **arrays)
    shutil.copy2(upstream_dir / SOMA_CORRECTIVES_ASSET, output_dir / SOMA_CORRECTIVES_ASSET)
    # The upstream meshes that map MHR, Anny, SMPL and SMPL-X bodies onto SOMA, copied unchanged.
    for spec in MODEL_TYPE_SPECS.values():
        if spec.asset_dir is None:
            continue
        (output_dir / spec.asset_dir).mkdir(exist_ok=True)
        for name in (spec.source_mesh_name, spec.target_mesh_name):
            shutil.copy2(upstream_dir / spec.asset_dir / name, output_dir / spec.asset_dir / name)


def _load_rig(stage: Any) -> dict[str, Any]:
    """The template skeleton with twist joints and the skin weights of its mid-LOD mesh."""
    skeleton = UsdSkel.Skeleton(next(prim for prim in stage.Traverse() if prim.IsA(UsdSkel.Skeleton)))
    joint_paths = [str(path) for path in skeleton.GetJointsAttr().Get()]
    path_index = {path: index for index, path in enumerate(joint_paths)}
    parents = np.asarray([path_index.get(path.rpartition("/")[0], 0) for path in joint_paths], dtype=np.int32)
    bind_pose_world = np.asarray(skeleton.GetBindTransformsAttr().Get(), dtype=np.float32).reshape(-1, 4, 4).mT
    t_pose_local = np.asarray(skeleton.GetRestTransformsAttr().Get(), dtype=np.float32).reshape(-1, 4, 4).mT
    bind_shape, _, skin_weights = _load_skin(stage, joint_paths, "c_skin_mid")
    return {
        "joint_paths": joint_paths,
        "joint_names": np.asarray([path.rpartition("/")[2] for path in joint_paths]),
        "joint_parent_ids": parents,
        "bind_pose_world": bind_pose_world,
        "t_pose_world": _forward_kinematics(t_pose_local, parents),
        "bind_shape": bind_shape,
        "skin_weights": skin_weights,
    }


def _public_rig(rig: dict[str, Any], public_names: list[str]) -> dict[str, Any]:
    """The rig without twist and helper joints; each removed joint's skin weights go to its nearest kept parent."""
    names = rig["joint_names"].tolist()
    keep = np.asarray([names.index(name) for name in public_names])
    parents = rig["joint_parent_ids"]
    new_index = {int(old): new for new, old in enumerate(keep)}
    public_parents = [new_index[_nearest_kept_parent(parents, int(old), new_index)] for old in keep]
    public_parents[0] = 0
    skin_weights = rig["skin_weights"].copy()
    for removed in sorted(set(range(len(names))) - set(new_index)):
        skin_weights[:, _nearest_kept_parent(parents, removed, new_index)] += skin_weights[:, removed]
    return {
        "joint_names": np.asarray(public_names),
        "joint_parent_ids": np.asarray(public_parents, dtype=np.int32),
        "bind_pose_world": rig["bind_pose_world"][keep],
        "t_pose_world": rig["t_pose_world"][keep],
        "bind_shape": rig["bind_shape"],
        "skin_weights": skin_weights[:, keep],
    }


def _procedural_arrays(
    procedural: dict[str, Any], rig: dict[str, Any], public_rig: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Twist-joint drivers: the public joints of each twist segment and the parameter matrices."""
    if procedural["rotation_extraction"] != "aligned_x_swing_twist":
        raise ValueError("SOMA procedural transforms must use aligned_x_swing_twist extraction")
    names = rig["joint_names"].tolist()
    public_names = public_rig["joint_names"].tolist()
    joint_index = {name: index for index, name in enumerate(names)}
    public_index = {name: index for index, name in enumerate(public_names)}
    public_full = {joint_index[name]: index for index, name in enumerate(public_names)}

    segments = procedural["segments"]
    start = np.asarray([public_index[segment["start_joint"]] for segment in segments])
    end = np.asarray([public_index[segment["end_joint"]] for segment in segments])
    # Without an explicit parent, a segment inherits twist from the nearest public ancestor of its start joint.
    parent_names = [
        segment.get("parent_joint")
        or names[_nearest_kept_parent(rig["joint_parent_ids"], joint_index[segment["start_joint"]], public_full)]
        for segment in segments
    ]
    twists = [(name, segment) for segment in segments for name in segment["twist_joints"]]
    twist_index = {name: index for index, (name, _) in enumerate(twists)}

    rotation_matrix = np.zeros((len(twists), len(public_names)), dtype=np.float32)
    for entry in procedural["parameter_matrices"]["rotation"]["entries"]:
        rotation_matrix[twist_index[entry["row"]], public_index[entry["column"]]] = entry["value"]
    # Translation entries replace the identity rows of the joints they place.
    translation_matrix = np.eye(len(names), dtype=np.float32)
    entries = procedural["parameter_matrices"]["translation"]["entries"]
    translation_matrix[[joint_index[entry["row"]] for entry in entries]] = 0.0
    for entry in entries:
        translation_matrix[joint_index[entry["row"]], joint_index[entry["column"]]] = entry["value"]

    return {
        "public_joint_indices_full": np.asarray([joint_index[name] for name in public_names], dtype=np.int64),
        "rotation_matrix": rotation_matrix,
        "translation_matrix": translation_matrix,
        "twist_joint_indices": np.asarray([joint_index[name] for name, _ in twists], dtype=np.int64),
        "twist_axis_ids": np.asarray(["xyz".index(segment["source_axis"].lower()) for _, segment in twists]),
        "twist_axis_signs": np.asarray([segment["source_sign"] for _, segment in twists], dtype=np.float32),
        "segment_start_joint_indices": start,
        "segment_end_joint_indices": end,
        "segment_parent_joint_indices": np.asarray([public_index[name] for name in parent_names]),
        "segment_reverse_indices": np.asarray(
            [index for index, segment in enumerate(segments) if segment.get("reverse", False)], dtype=np.int64
        ),
        "segment_alignment_rotations": _alignment_rotations(public_rig["bind_pose_world"], start, end),
    }


def _rig_arrays(rig: dict[str, Any], prefix: str) -> dict[str, np.ndarray]:
    parents = rig["joint_parent_ids"]
    arrays = {
        "joint_names": rig["joint_names"],
        "joint_parent_ids": parents,
        "bind_pose_world": rig["bind_pose_world"],
        "bind_pose_local": _world_to_local(rig["bind_pose_world"], parents),
        "t_pose_world": rig["t_pose_world"],
        "t_pose_local": _world_to_local(rig["t_pose_world"], parents),
        "bind_shape": rig["bind_shape"],
    } | _sparse_arrays(rig["skin_weights"], "skinning_weights")
    return {f"{prefix}{name}": value for name, value in arrays.items()}


def _load_skin(
    stage: Any, joint_paths: list[str], mesh_name: str
) -> tuple[Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"], Float[np.ndarray, "V J"]]:
    """Bind-pose points, fan-triangulated faces and dense skin weights of a skeleton-bound mesh."""
    prim = next(prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh) and prim.GetName() == mesh_name)
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    face_counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    face_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    faces = np.split(face_indices, np.cumsum(face_counts)[:-1])
    triangles = np.asarray([(face[0], face[i], face[i + 1]) for face in faces for i in range(1, len(face) - 1)])

    binding = UsdSkel.BindingAPI(prim)
    path_index = {path: index for index, path in enumerate(joint_paths)}
    binding_joints = np.asarray([path_index[str(path)] for path in binding.GetJointsAttr().Get()])
    influences = binding.GetJointIndicesPrimvar().GetElementSize()
    joints = binding_joints[np.asarray(binding.GetJointIndicesPrimvar().Get()).reshape(-1, influences)]
    weights = np.asarray(binding.GetJointWeightsPrimvar().Get(), dtype=np.float32).reshape(-1, influences)
    skin_weights = np.zeros((len(points), len(joint_paths)), dtype=np.float32)
    np.add.at(skin_weights, (np.arange(len(points))[:, None], joints), weights)
    return points, triangles.astype(np.int64), skin_weights


def _sparse_arrays(dense: Float[np.ndarray, "V J"], name: str) -> dict[str, np.ndarray]:
    sparse = csc_matrix(dense)
    return {
        f"{name}_data": sparse.data.astype(np.float32),
        f"{name}_indices": sparse.indices.astype(np.int32),
        f"{name}_indptr": sparse.indptr.astype(np.int32),
        f"{name}_shape": np.asarray(sparse.shape, dtype=np.int32),
    }


def _alignment_rotations(
    bind_pose_world: Float[np.ndarray, "J 4 4"], start: Int[np.ndarray, "S"], end: Int[np.ndarray, "S"]
) -> Float[np.ndarray, "S 3 3"]:
    """Segment frames whose x axis runs along the bone from start to end, with y closest to the start joint's y."""
    start_rotation = bind_pose_world[start, :3, :3]
    span = _normalize(bind_pose_world[end, :3, 3] - bind_pose_world[start, :3, 3])
    x_axis = span * np.where(np.sum(start_rotation[..., 0] * span, axis=-1, keepdims=True) >= 0.0, 1.0, -1.0)
    y_candidate = start_rotation[..., 1]
    y_axis = _normalize(y_candidate - np.sum(y_candidate * x_axis, axis=-1, keepdims=True) * x_axis)
    z_axis = _normalize(np.cross(x_axis, y_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return np.stack((x_axis, y_axis, z_axis), axis=-1).astype(np.float32)


def _nearest_kept_parent(parents: Int[np.ndarray, "J"], joint: int, kept: dict[int, int]) -> int:
    """Closest ancestor of ``joint`` among the keys of ``kept``."""
    parent = int(parents[joint])
    while parent not in kept and parent != int(parents[parent]):
        parent = int(parents[parent])
    return parent


def _nearest_vertex_ids(source: Float[np.ndarray, "Vs 3"], target: Float[np.ndarray, "Vt 3"]) -> Int[np.ndarray, "Vt"]:
    chunks = np.array_split(target, max(1, len(target) // 128))
    return np.concatenate([np.argmin(np.sum((chunk[:, None] - source) ** 2, axis=-1), axis=1) for chunk in chunks])


def _forward_kinematics(local: Float[np.ndarray, "J 4 4"], parents: Int[np.ndarray, "J"]) -> Float[np.ndarray, "J 4 4"]:
    world = local.copy()
    for joint, parent in enumerate(parents.tolist()):
        if parent != joint:
            world[joint] = world[parent] @ local[joint]
    return world


def _world_to_local(world: Float[np.ndarray, "J 4 4"], parents: Int[np.ndarray, "J"]) -> Float[np.ndarray, "J 4 4"]:
    local = np.linalg.inv(world[parents]) @ world
    local[0] = world[0]
    return local.astype(np.float32)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


if __name__ == "__main__":
    typer.run(main)
