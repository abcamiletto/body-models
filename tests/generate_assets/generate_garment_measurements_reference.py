"""Generate the private GarmentMeasurements rig asset from upstream data.

Run this with Blender, not with ``uv run``:

    blender --background --python tests/generate_assets/generate_garment_measurements_reference.py -- \
        /path/to/GarmentMeasurements/data tests/assets/garment_measurements/model

The generated ``garment_measurements.npz`` is the only asset loaded by the runtime
model. Blender is used only here to read upstream ``template/male.fbx``.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream_data", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)

    upstream_data = args.upstream_data
    mean_vertices, components, eigenvalues = _load_pca(upstream_data / "pca" / "point.pca")
    _, faces = _load_obj(upstream_data / "pca" / "mean.obj")
    rig = _load_fbx_rig(upstream_data / "template" / "male.fbx", mean_vertices.shape[0])
    mvc_weights = _compute_mvc_weights(mean_vertices, faces, rig["joint_positions"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "garment_measurements.npz",
        mean_vertices=mean_vertices.astype(np.float32),
        components=components.astype(np.float32),
        eigenvalues=eigenvalues.astype(np.float32),
        faces=faces.astype(np.int64),
        joint_names=np.asarray(rig["joint_names"]),
        parents=rig["parents"].astype(np.int64),
        bind_quats=rig["bind_quats"].astype(np.float32),
        skin_weights=rig["skin_weights"].astype(np.float32),
        mvc_weights=mvc_weights.astype(np.float32),
    )


def _load_pca(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = path.read_bytes()
    dimension, num_components = struct.unpack_from("<II", data, 0)
    offset = 8
    matrix_count = dimension * num_components
    matrix = np.frombuffer(data, dtype="<f8", count=matrix_count, offset=offset).reshape(
        (dimension, num_components), order="F"
    )
    offset += matrix_count * 8
    mean = np.frombuffer(data, dtype="<f8", count=dimension, offset=offset)
    offset += dimension * 8
    eigenvalues = np.frombuffer(data, dtype="<f8", count=num_components, offset=offset)
    return mean.reshape(-1, 3), matrix.reshape(-1, 3, num_components), eigenvalues


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            face = [int(token.split("/", 1)[0]) - 1 for token in line.split()[1:]]
            if len(face) != 3:
                raise ValueError(f"Expected triangular faces in {path}")
            faces.append(face)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _load_fbx_rig(path: Path, num_vertices: int) -> dict[str, np.ndarray | list[str]]:
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.import_scene.fbx(filepath=str(path))

    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.vertex_groups]
    if len(armatures) != 1:
        raise ValueError(f"Expected one FBX armature, found {len(armatures)}")
    if not meshes:
        raise ValueError("Expected at least one skinned mesh in FBX")

    mesh = next((obj for obj in meshes if obj.name in {"skin", "H_DDS_HighResShape"}), meshes[0])
    if len(mesh.data.vertices) != num_vertices:
        raise ValueError(f"FBX mesh has {len(mesh.data.vertices)} vertices, expected {num_vertices}")

    armature = armatures[0]
    bones = list(armature.data.bones)
    bone_index = {bone.name: index for index, bone in enumerate(bones)}
    joint_names = [bone.name for bone in bones]
    parents = np.asarray([bone_index[bone.parent.name] if bone.parent else -1 for bone in bones], dtype=np.int64)

    world_mats = [armature.matrix_world @ bone.matrix_local for bone in bones]
    bind_quats = np.zeros((len(bones), 4), dtype=np.float64)
    joint_positions = np.zeros((len(bones), 3), dtype=np.float64)
    for index, bone in enumerate(bones):
        local = world_mats[index] if bone.parent is None else world_mats[parents[index]].inverted() @ world_mats[index]
        quat = local.to_quaternion()
        bind_quats[index] = [quat.w, quat.x, quat.y, quat.z]
        joint_positions[index] = world_mats[index].translation[:]

    skin_weights = np.zeros((num_vertices, len(bones)), dtype=np.float64)
    group_to_joint = {group.index: bone_index[group.name] for group in mesh.vertex_groups if group.name in bone_index}
    for vertex in mesh.data.vertices:
        for group in vertex.groups:
            joint_index = group_to_joint.get(group.group)
            if joint_index is not None:
                skin_weights[vertex.index, joint_index] = group.weight

    row_sums = skin_weights.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("FBX skinning has vertices with no joint weights")
    skin_weights = skin_weights / row_sums

    return {
        "joint_names": joint_names,
        "parents": parents,
        "bind_quats": bind_quats,
        "joint_positions": joint_positions,
        "skin_weights": skin_weights,
    }


def _compute_mvc_weights(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
    rings, boundary = _ordered_vertex_rings(faces, len(vertices))
    weights = np.zeros((len(vertices), len(points)), dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for point_index, point in enumerate(points):
        for vertex_index, vertex in enumerate(vertices):
            if boundary[vertex_index] or not rings[vertex_index]:
                continue
            vi = vertex - point
            norm_vi = np.linalg.norm(vi)
            if norm_vi < eps:
                weights[vertex_index, point_index] = 1.0
                continue

            ei = vi / norm_vi
            wi = 0.0
            ring = rings[vertex_index]
            for ring_index, j in enumerate(ring):
                k = ring[(ring_index + 1) % len(ring)]
                ej = _safe_unit(vertices[j] - point)
                ek = _safe_unit(vertices[k] - point)
                njk = _safe_unit(np.cross(ej, ek))
                nij = _safe_unit(np.cross(ei, ej))
                nki = _safe_unit(np.cross(ek, ei))
                beta_jk = abs(np.arccos(np.clip(np.dot(ej, ek), -1.0, 1.0)))
                beta_ij = abs(np.arccos(np.clip(np.dot(ei, ej), -1.0, 1.0)))
                beta_ki = abs(np.arccos(np.clip(np.dot(ek, ei), -1.0, 1.0)))
                denom = 2.0 * np.dot(ei, njk)
                if abs(denom) > eps:
                    wi += (beta_jk + beta_ij * np.dot(nij, njk) + beta_ki * np.dot(nki, njk)) / denom
            weights[vertex_index, point_index] = wi / norm_vi

        total = weights[:, point_index].sum()
        if abs(total) > eps:
            weights[:, point_index] /= total

    return weights


def _ordered_vertex_rings(faces: np.ndarray, num_vertices: int) -> tuple[list[list[int]], np.ndarray]:
    next_neighbor = [{} for _ in range(num_vertices)]
    boundary_edges: set[tuple[int, int]] = set()
    edge_counts: dict[tuple[int, int], int] = {}

    for a, b, c in faces:
        next_neighbor[a][c] = b
        next_neighbor[b][a] = c
        next_neighbor[c][b] = a
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (min(u, v), max(u, v))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    for edge, count in edge_counts.items():
        if count == 1:
            boundary_edges.add(edge)

    boundary = np.zeros(num_vertices, dtype=bool)
    for u, v in boundary_edges:
        boundary[u] = True
        boundary[v] = True

    rings: list[list[int]] = []
    for vertex, mapping in enumerate(next_neighbor):
        if boundary[vertex] or not mapping:
            rings.append([])
            continue
        start = next(iter(mapping))
        ring = [start]
        current = mapping[start]
        while current != start:
            if current not in mapping or current in ring:
                ring = []
                boundary[vertex] = True
                break
            ring.append(current)
            current = mapping[current]
        rings.append(ring)
    return rings, boundary


def _safe_unit(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value)
    if norm <= np.finfo(np.float64).eps:
        return np.zeros(3, dtype=np.float64)
    return value / norm


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:])
