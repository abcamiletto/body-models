"""Generate symmetric, globally optimized LODs for a rigid SMPL mannequin."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymeshlab
import trimesh
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]
from trimesh.exchange.obj import export_obj

from body_models.smpl_humanoid import SmplMannequin

DEFAULT_BUDGETS = (40_000, 15_000, 5_000)
CANDIDATE_RATIOS = (1.0, 0.65, 0.45, 0.32, 0.23, 0.16, 0.11, 0.075, 0.05, 0.032, 0.02)
SAMPLE_COUNT = 1_200
MIRROR = np.diag((-1.0, 1.0, 1.0))


@dataclass
class Candidate:
    mesh: trimesh.Trimesh
    cost: int
    error: float
    method: str


@dataclass
class Part:
    key: str
    indices: tuple[int, ...]
    source: trimesh.Trimesh
    candidates: list[Candidate]
    importance: float

    @property
    def paired(self) -> bool:
        return len(self.indices) == 2


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    robot = SmplMannequin(model_path=source_path)
    transforms = np.asarray(robot.forward_links(**robot.get_tpose()))
    parts = build_parts(robot, transforms)
    for index, part in enumerate(parts, start=1):
        part.candidates = generate_candidates(part)
        print(f"[{index:02d}/{len(parts)}] {part.key}: {len(part.candidates)} candidates")

    manifest: dict[str, Any] = {
        "source": source_path.name,
        "source_vertices": int(robot._weights.vertices.shape[0]),
        "source_faces": int(robot.faces.shape[0]),
        "lods": {},
    }
    for level, budget in enumerate(args.budgets):
        selection = allocate_budget(parts, budget)
        lod_name = f"lod{level}_{budget // 1000}k"
        lod_path = output / lod_name
        write_lod(source_path, lod_path, robot, transforms, parts, selection)
        report = validate_lod(source_path, lod_path / source_path.name)
        report["target_vertices"] = budget
        report["parts"] = {
            part.key: {
                "vertices": part.candidates[candidate_index].cost,
                "error": part.candidates[candidate_index].error,
                "method": part.candidates[candidate_index].method,
            }
            for part, candidate_index in zip(parts, selection, strict=True)
        }
        (lod_path / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        manifest["lods"][lod_name] = report
        print(
            f"{lod_name}: {report['vertices']:,} vertices, "
            f"p99 {report['surface_error_mm']['p99']:.3f} mm, "
            f"symmetry {report['symmetry_max_error_mm']:.6f} mm"
        )

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build_parts(robot: SmplMannequin, transforms: np.ndarray) -> list[Part]:
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(robot.link_names):
        groups.setdefault(symmetry_key(name), []).append(index)

    parts = []
    for key, grouped_indices in groups.items():
        if len(grouped_indices) == 2:
            left = next(index for index in grouped_indices if robot.link_names[index].startswith("L_"))
            right = next(index for index in grouped_indices if robot.link_names[index].startswith("R_"))
            indices = (left, right)
            source = world_mesh(robot, left, transforms)
        elif len(grouped_indices) == 1:
            indices = (grouped_indices[0],)
            source = world_mesh(robot, grouped_indices[0], transforms)
        else:
            raise ValueError(f"Unexpected symmetry group {key}: {grouped_indices}")
        parts.append(
            Part(
                key=key,
                indices=indices,
                source=source,
                candidates=[],
                importance=part_importance(key, source),
            )
        )
    return parts


def generate_candidates(part: Part) -> list[Candidate]:
    candidates = []
    if part.paired:
        input_mesh = part.source
        for ratio in CANDIDATE_RATIOS:
            mesh = input_mesh.copy() if ratio == 1.0 else simplify(input_mesh, ratio, preserve_boundary=False)
            candidates.append(candidate(part, mesh, f"qem:{ratio:g}"))
        if "ball_" in part.key:
            for subdivisions in (1, 2, 3):
                mesh = fitted_icosphere(input_mesh, subdivisions)
                candidates.append(candidate(part, mesh, f"icosphere:{subdivisions}"))
    else:
        half = positive_half(part.source)
        for ratio in CANDIDATE_RATIOS:
            simplified = half.copy() if ratio == 1.0 else simplify(half, ratio, preserve_boundary=True)
            candidates.append(candidate(part, mirror_center(simplified), f"symmetric-qem:{ratio:g}"))
    return pareto_candidates(candidates)


def candidate(part: Part, mesh: trimesh.Trimesh, method: str) -> Candidate:
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        raise ValueError(f"{part.key} produced a non-watertight candidate with {method}")
    cost = len(mesh.vertices) * (2 if part.paired else 1)
    seed = zlib.crc32(f"{part.key}:{method}".encode())
    error = mesh_error(part.source, mesh, seed=seed)
    return Candidate(mesh=mesh, cost=cost, error=error, method=method)


def simplify(mesh: trimesh.Trimesh, ratio: float, *, preserve_boundary: bool) -> trimesh.Trimesh:
    target_faces = max(4, round(len(mesh.faces) * ratio))
    quality = curvature_quality(mesh)
    pmesh = pymeshlab.Mesh(  # ty: ignore[unresolved-attribute]
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        v_normals_matrix=np.asarray(mesh.vertex_normals, dtype=np.float64),
        v_scalar_array=quality,
    )
    mesh_set = pymeshlab.MeshSet()  # ty: ignore[unresolved-attribute]
    mesh_set.add_mesh(pmesh)
    mesh_set.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=target_faces,
        qualitythr=0.5,
        preserveboundary=preserve_boundary,
        boundaryweight=20.0,
        preservenormal=True,
        preservetopology=True,
        optimalplacement=True,
        planarquadric=True,
        planarweight=0.001,
        qualityweight=True,
        autoclean=True,
    )
    result = mesh_set.current_mesh()
    simplified = trimesh.Trimesh(
        vertices=result.vertex_matrix(),
        faces=result.face_matrix(),
        process=False,
    )
    simplified.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(simplified)
    if preserve_boundary:
        simplified.vertices[np.abs(simplified.vertices[:, 0]) < 1e-8, 0] = 0.0
    return simplified


def curvature_quality(mesh: trimesh.Trimesh) -> np.ndarray:
    normals = np.asarray(mesh.vertex_normals)
    quality = np.zeros(len(normals), dtype=np.float64)
    for index, neighbors in enumerate(mesh.vertex_neighbors):
        if neighbors:
            alignment = normals[np.asarray(neighbors)] @ normals[index]
            quality[index] = np.mean(1.0 - np.clip(alignment, -1.0, 1.0))
    maximum = quality.max(initial=0.0)
    if maximum > 0.0:
        quality /= maximum
    return np.maximum(quality, 0.02)


def positive_half(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    half = trimesh.intersections.slice_mesh_plane(
        mesh,
        plane_normal=(1.0, 0.0, 0.0),
        plane_origin=(0.0, 0.0, 0.0),
        cap=False,
    )
    half.vertices[np.abs(half.vertices[:, 0]) < 1e-8, 0] = 0.0
    return half


def mirror_center(half: trimesh.Trimesh) -> trimesh.Trimesh:
    mirrored = half.copy()
    mirrored.vertices = mirrored.vertices @ MIRROR
    mirrored.faces = mirrored.faces[:, ::-1]
    mesh = trimesh.util.concatenate((half, mirrored))
    mesh.merge_vertices(digits_vertex=10)
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    return mesh


def fitted_icosphere(source: trimesh.Trimesh, subdivisions: int) -> trimesh.Trimesh:
    center = source.vertices.mean(axis=0)
    radius = np.median(np.linalg.norm(source.vertices - center, axis=1))
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.apply_translation(center)
    return mesh


def mesh_error(source: trimesh.Trimesh, candidate_mesh: trimesh.Trimesh, *, seed: int) -> float:
    metrics = surface_metrics(source, candidate_mesh, sample_count=SAMPLE_COUNT, seed=seed)
    scale = np.linalg.norm(source.extents)
    distance_error = metrics["rms_distance"] / max(scale, 1e-8)
    normal_error = np.deg2rad(metrics["rms_normal_degrees"])
    return float(distance_error + 0.015 * normal_error)


def surface_metrics(
    source: trimesh.Trimesh,
    candidate_mesh: trimesh.Trimesh,
    *,
    sample_count: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    source_points, source_faces = trimesh.sample.sample_surface(source, sample_count, seed=seed)
    candidate_points, candidate_faces = trimesh.sample.sample_surface(candidate_mesh, sample_count, seed=seed + 1)
    closest_candidate, source_distances, candidate_face_indices = trimesh.proximity.closest_point(
        candidate_mesh,
        source_points,
    )
    closest_source, candidate_distances, source_face_indices = trimesh.proximity.closest_point(
        source,
        candidate_points,
    )
    del closest_candidate, closest_source

    source_normals = source.face_normals[source_faces]
    candidate_normals = candidate_mesh.face_normals[candidate_faces]
    nearest_candidate_normals = candidate_mesh.face_normals[candidate_face_indices]
    nearest_source_normals = source.face_normals[source_face_indices]
    cosines = np.concatenate(
        (
            np.einsum("ij,ij->i", source_normals, nearest_candidate_normals),
            np.einsum("ij,ij->i", candidate_normals, nearest_source_normals),
        )
    )
    normal_angles = np.rad2deg(np.arccos(np.clip(cosines, -1.0, 1.0)))
    distances = np.concatenate((source_distances, candidate_distances))
    return {
        "distances": distances,
        "normal_angles": normal_angles,
        "rms_distance": float(np.sqrt(np.mean(np.square(distances)))),
        "rms_normal_degrees": float(np.sqrt(np.mean(np.square(normal_angles)))),
    }


def pareto_candidates(candidates: list[Candidate]) -> list[Candidate]:
    by_cost = {}
    for item in candidates:
        previous = by_cost.get(item.cost)
        if previous is None or item.error < previous.error:
            by_cost[item.cost] = item
    frontier = []
    best_error = np.inf
    for item in sorted(by_cost.values(), key=lambda value: value.cost):
        if item.error < best_error:
            frontier.append(item)
            best_error = item.error
    return frontier


def allocate_budget(parts: list[Part], budget: int) -> list[int]:
    costs = [[candidate.cost for candidate in part.candidates] for part in parts]
    dp = np.full(budget + 1, np.inf)
    dp[0] = 0.0
    choices = []
    for part in parts:
        next_dp = np.full_like(dp, np.inf)
        choice = np.full(budget + 1, -1, dtype=np.int16)
        for candidate_index, item in enumerate(part.candidates):
            if item.cost > budget:
                continue
            values = dp[: budget + 1 - item.cost] + part.importance * item.error
            target = next_dp[item.cost :]
            better = values < target
            target[better] = values[better]
            choice[item.cost :][better] = candidate_index
        if not np.isfinite(next_dp).any():
            raise ValueError(f"Vertex budget {budget:,} is too small.")
        dp = next_dp
        choices.append(choice)

    used = int(np.nanargmin(dp))
    selection = [0] * len(parts)
    for part_index in range(len(parts) - 1, -1, -1):
        candidate_index = int(choices[part_index][used])
        selection[part_index] = candidate_index
        used -= costs[part_index][candidate_index]
    return selection


def write_lod(
    source_path: Path,
    output: Path,
    robot: SmplMannequin,
    transforms: np.ndarray,
    parts: list[Part],
    selection: list[int],
) -> None:
    assets = output / f"{source_path.stem}_assets"
    if output.exists():
        shutil.rmtree(output)
    assets.mkdir(parents=True)
    shutil.copy2(source_path, output / source_path.name)

    for part, candidate_index in zip(parts, selection, strict=True):
        selected = part.candidates[candidate_index].mesh
        if part.paired:
            left, right = part.indices
            write_local_mesh(selected, transforms[left], assets / f"{robot.link_names[left]}.obj")
            mirrored = selected.copy()
            mirrored.vertices = mirrored.vertices @ MIRROR
            mirrored.faces = mirrored.faces[:, ::-1]
            write_local_mesh(mirrored, transforms[right], assets / f"{robot.link_names[right]}.obj")
        else:
            center = part.indices[0]
            write_local_mesh(selected, transforms[center], assets / f"{robot.link_names[center]}.obj")


def write_local_mesh(world: trimesh.Trimesh, transform: np.ndarray, path: Path) -> None:
    vertices = (world.vertices - transform[:3, 3]) @ transform[:3, :3]
    mesh = trimesh.Trimesh(vertices=vertices, faces=world.faces, process=False)
    text = export_obj(
        mesh,
        include_normals=False,
        include_color=False,
        include_texture=False,
    )
    path.write_text(text)


def validate_lod(source_path: Path, lod_path: Path) -> dict:
    source = SmplMannequin(model_path=source_path)
    lod = SmplMannequin(model_path=lod_path)
    source_transforms = np.asarray(source.forward_links(**source.get_tpose()))
    lod_transforms = np.asarray(lod.forward_links(**lod.get_tpose()))
    if source.joint_names != lod.joint_names:
        raise ValueError("LOD changed the joint hierarchy.")

    symmetry_error = validate_symmetry(lod, lod_transforms)
    watertight = all(world_mesh(lod, index, lod_transforms).is_watertight for index in range(len(lod.link_names)))
    if not watertight:
        raise ValueError(f"{lod_path} contains a non-watertight mesh.")

    distances = []
    normal_angles = []
    for index in range(len(source.link_names)):
        metrics = surface_metrics(
            world_mesh(source, index, source_transforms),
            world_mesh(lod, index, lod_transforms),
            sample_count=600,
            seed=10_000 + index,
        )
        distances.append(metrics["distances"])
        normal_angles.append(metrics["normal_angles"])
    distances = np.concatenate(distances) * 1_000.0
    normal_angles = np.concatenate(normal_angles)

    poses = (
        source.get_tpose()["body_pose"],
        source.get_apose()["body_pose"],
        np.linspace(-0.25, 0.25, source.num_actuated, dtype=np.float32),
    )
    fk_error = max(
        float(np.max(np.abs(source.forward_links(body_pose=pose) - lod.forward_links(body_pose=pose))))
        for pose in poses
    )
    return {
        "vertices": int(lod._weights.vertices.shape[0]),
        "faces": int(lod.faces.shape[0]),
        "watertight": watertight,
        "symmetry_max_error_mm": symmetry_error * 1_000.0,
        "fk_max_error": fk_error,
        "surface_error_mm": {
            "mean": float(np.mean(distances)),
            "rms": float(np.sqrt(np.mean(np.square(distances)))),
            "p95": float(np.percentile(distances, 95)),
            "p99": float(np.percentile(distances, 99)),
            "max": float(np.max(distances)),
        },
        "normal_error_degrees": {
            "mean": float(np.mean(normal_angles)),
            "p95": float(np.percentile(normal_angles, 95)),
            "p99": float(np.percentile(normal_angles, 99)),
        },
    }


def validate_symmetry(robot: SmplMannequin, transforms: np.ndarray) -> float:
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(robot.link_names):
        groups.setdefault(symmetry_key(name), []).append(index)
    maximum = 0.0
    for grouped_indices in groups.values():
        if len(grouped_indices) == 2:
            left = next(index for index in grouped_indices if robot.link_names[index].startswith("L_"))
            right = next(index for index in grouped_indices if robot.link_names[index].startswith("R_"))
            left_vertices = world_mesh(robot, left, transforms).vertices
            right_vertices = world_mesh(robot, right, transforms).vertices @ MIRROR
        else:
            center = grouped_indices[0]
            left_vertices = world_mesh(robot, center, transforms).vertices
            right_vertices = left_vertices @ MIRROR
        maximum = max(maximum, symmetric_set_error(left_vertices, right_vertices))
    if maximum > 1e-7:
        raise ValueError(f"LOD is not symmetric; maximum vertex error is {maximum:g} m.")
    return maximum


def symmetric_set_error(first: np.ndarray, second: np.ndarray) -> float:
    forward = cKDTree(first).query(second)[0].max(initial=0.0)
    reverse = cKDTree(second).query(first)[0].max(initial=0.0)
    return float(max(forward, reverse))


def world_mesh(robot: SmplMannequin, index: int, transforms: np.ndarray) -> trimesh.Trimesh:
    vertex_start = robot.link_vertex_starts[index]
    vertex_count = robot.link_vertex_counts[index]
    face_start = robot.link_face_starts[index]
    face_count = robot.link_face_counts[index]
    vertices = robot._weights.vertices[vertex_start : vertex_start + vertex_count]
    faces = robot.faces[face_start : face_start + face_count] - vertex_start
    transform = transforms[index]
    vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def symmetry_key(name: str) -> str:
    key = re.sub(r"(^|_)[LR]_", r"\1S_", name)
    key = re.sub(r"_J\d+__", "_J__", key)
    return re.sub(r"_\d+_\d+$", "", key)


def part_importance(key: str, mesh: trimesh.Trimesh) -> float:
    importance = np.sqrt(mesh.area)
    if any(name in key for name in ("head", "palm", "chest_shell", "pelvis_shell")):
        importance *= 2.0
    elif any(name in key for name in ("forearm", "shin", "thigh", "upper_arm", "rear_foot")):
        importance *= 1.35
    return float(importance)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Full-resolution generated mannequin XML.")
    parser.add_argument("output", type=Path, help="Output directory for all LODs.")
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=DEFAULT_BUDGETS,
        help="Total vertex budgets, ordered from highest to lowest detail.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
