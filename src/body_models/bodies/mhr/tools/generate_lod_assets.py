# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "bpy>=4.2.0",
#   "jaxtyping>=0.2.28",
#   "numpy>=1.26,<2",
#   "typer>=0.9.0",
# ]
# ///
"""Generate MHR LOD mesh assets from the upstream FBX files.

Run this as a self-contained PEP 723 script:

    uv run --python 3.11 --no-project src/body_models/bodies/mhr/tools/generate_lod_assets.py \
        /path/to/mhr/assets /path/to/output
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from jaxtyping import Float, Int

LODS = tuple(range(7))


def main(
    asset_dir: Annotated[Path, typer.Argument()],
    output_dir: Annotated[Path, typer.Argument()],
    lod: Annotated[list[int] | None, typer.Option("--lod")] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for lod_value in lod or LODS:
        if lod_value not in LODS:
            raise ValueError(f"MHR lod must be one of {LODS}, got {lod_value}")
        fbx_path = asset_dir / f"lod{lod_value}.fbx"
        output_path = output_dir / f"mhr_lod{lod_value}.npz"
        print(f"MHR: generating {output_path} from {fbx_path}", flush=True)
        save_lod_asset(fbx_path, output_path)


def save_lod_asset(fbx_path: Path, output_path: Path) -> None:
    bpy = importlib.import_module("bpy")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.vertex_groups]
    if len(meshes) != 1:
        raise ValueError(f"Expected one skinned mesh in {fbx_path}, found {len(meshes)}")

    mesh = meshes[0]
    faces = _faces(mesh, fbx_path)
    base_vertices = np.asarray([vertex.co[:] for vertex in mesh.data.vertices], dtype=np.float32)
    blendshape_dirs = _blendshape_dirs(mesh, base_vertices, fbx_path)
    vertex_indices, joint_indices, skin_weights, joint_names = _sparse_skinning(mesh)

    np.savez_compressed(
        output_path,
        base_vertices=base_vertices,
        blendshape_dirs=blendshape_dirs,
        faces=faces,
        skin_vertex_indices=vertex_indices,
        skin_joint_indices=joint_indices,
        skin_weights=skin_weights,
        skin_joint_names=np.asarray(joint_names),
    )


def _faces(mesh, fbx_path: Path) -> Int[np.ndarray, "F 3"]:
    faces = [[vertex for vertex in polygon.vertices] for polygon in mesh.data.polygons]
    if any(len(face) != 3 for face in faces):
        raise ValueError(f"Expected only triangular faces in {fbx_path}")
    return np.asarray(faces, dtype=np.int64)


def _blendshape_dirs(
    mesh,
    base_vertices: Float[np.ndarray, "V 3"],
    fbx_path: Path,
) -> Float[np.ndarray, "117 V 3"]:
    keys = mesh.data.shape_keys.key_blocks if mesh.data.shape_keys else []
    key_names = [key.name for key in keys]
    expected = ["Basis", *(f"shape_{index}" for index in range(117))]
    if key_names != expected:
        raise ValueError(f"{fbx_path} has shape keys {key_names}, expected {expected}")

    deltas = []
    for key in keys[1:]:
        vertices = np.asarray([point.co[:] for point in key.data], dtype=np.float32)
        deltas.append(vertices - base_vertices)
    return np.asarray(deltas, dtype=np.float32)


def _sparse_skinning(
    mesh,
) -> tuple[Int[np.ndarray, "N"], Int[np.ndarray, "N"], Float[np.ndarray, "N"], list[str]]:
    group_names = [group.name for group in mesh.vertex_groups]
    vertex_indices: list[int] = []
    joint_indices: list[int] = []
    skin_weights: list[float] = []

    for vertex in mesh.data.vertices:
        for group in vertex.groups:
            if group.weight <= 0:
                continue
            vertex_indices.append(vertex.index)
            joint_indices.append(group.group)
            skin_weights.append(group.weight)

    counts = np.bincount(np.asarray(vertex_indices), minlength=len(mesh.data.vertices))
    if np.any(counts == 0):
        raise ValueError("MHR FBX skinning has vertices with no joint weights")

    return (
        np.asarray(vertex_indices, dtype=np.int64),
        np.asarray(joint_indices, dtype=np.int64),
        np.asarray(skin_weights, dtype=np.float32),
        group_names,
    )


if __name__ == "__main__":
    typer.run(main)
