"""Procedural assets for the rigid SMPL humanoid model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from jaxtyping import Float, Int

from body_models.robots.smpl_humanoid.constants import BODY_JOINTS, JOINT_NAMES, PARENTS

Array = Any
PathLike = Path | str

ROOT_QPOS_SIZE = 7
ROOT_QVEL_SIZE = 6
JOINT_POS_SIZE = 3 * len(BODY_JOINTS)
BODY_COUNT = len(JOINT_NAMES)
QPOS_SIZE = ROOT_QPOS_SIZE + JOINT_POS_SIZE
QVEL_SIZE = ROOT_QVEL_SIZE + JOINT_POS_SIZE
ACTION_SIZE = JOINT_POS_SIZE

LOCAL_OFFSETS = np.array(
    [
        [0.000, 0.000, 0.000],
        [0.090, 0.000, -0.090],
        [-0.090, 0.000, -0.090],
        [0.000, 0.000, 0.120],
        [0.010, 0.000, -0.410],
        [-0.010, 0.000, -0.410],
        [0.000, 0.000, 0.150],
        [0.000, 0.000, -0.400],
        [0.000, 0.000, -0.400],
        [0.000, 0.000, 0.160],
        [0.000, 0.090, -0.060],
        [0.000, 0.090, -0.060],
        [0.000, 0.000, 0.130],
        [0.075, 0.000, 0.110],
        [-0.075, 0.000, 0.110],
        [0.000, 0.000, 0.120],
        [0.170, 0.000, 0.000],
        [-0.170, 0.000, 0.000],
        [0.270, 0.000, 0.000],
        [-0.270, 0.000, 0.000],
        [0.250, 0.000, 0.000],
        [-0.250, 0.000, 0.000],
        [0.100, 0.000, 0.000],
        [-0.100, 0.000, 0.000],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class SmplHumanoidWeights:
    joint_names: list[str]
    parents: list[int]
    local_offsets: Float[Array, "J 3"]
    rest_local_rotations: Float[Array, "J 3 3"]
    vertices: Float[Array, "V 3"]
    faces: Int[Array, "F 3"]
    link_joint_indices: list[int]
    link_vertex_starts: list[int]
    link_vertex_counts: list[int]
    link_face_starts: list[int]
    link_face_counts: list[int]
    link_geom_positions: Float[Array, "L 3"]
    link_geom_rotations: Float[Array, "L 3 3"]
    link_names: list[str]
    qpos_joint_indices: list[int]
    qpos_joint_axes: Float[Array, "Q 3"]
    qpos_joint_limits: Float[Array, "Q 2"]
    qpos_joint_names: list[str]


def load_model_data(model_path: PathLike | None = None, *, dtype=np.float32) -> SmplHumanoidWeights:
    """Build a lightweight rigid SMPL humanoid from procedural link meshes."""
    if model_path is not None:
        return _load_xml_model_data(Path(model_path), dtype=dtype)

    vertices, faces, link_data = _build_link_meshes(dtype=dtype)
    return _weights_from_parts(
        local_offsets=LOCAL_OFFSETS.astype(dtype),
        rest_local_rotations=np.repeat(np.eye(3, dtype=dtype)[None], len(JOINT_NAMES), axis=0),
        vertices=vertices,
        faces=faces,
        link_data=link_data,
        dtype=dtype,
    )


def _weights_from_parts(
    *,
    local_offsets: np.ndarray,
    rest_local_rotations: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    link_data: dict[str, Any],
    dtype,
) -> SmplHumanoidWeights:
    by_name = {name: i for i, name in enumerate(JOINT_NAMES)}
    qpos_joint_indices = [by_name[name] for name, _ in BODY_JOINTS]
    qpos_joint_names = [name for name, _ in BODY_JOINTS]
    qpos_joint_limits = np.repeat(np.array([[-np.pi, np.pi]], dtype=dtype), len(BODY_JOINTS), axis=0)
    return SmplHumanoidWeights(
        joint_names=JOINT_NAMES.copy(),
        parents=PARENTS.copy(),
        local_offsets=local_offsets.astype(dtype),
        rest_local_rotations=rest_local_rotations.astype(dtype),
        vertices=vertices.astype(dtype),
        faces=faces.astype(np.int64),
        link_joint_indices=link_data["joint_indices"],
        link_vertex_starts=link_data["vertex_starts"],
        link_vertex_counts=link_data["vertex_counts"],
        link_face_starts=link_data["face_starts"],
        link_face_counts=link_data["face_counts"],
        link_geom_positions=link_data["geom_positions"].astype(dtype),
        link_geom_rotations=link_data["geom_rotations"].astype(dtype),
        link_names=link_data["names"],
        qpos_joint_indices=qpos_joint_indices,
        qpos_joint_axes=np.zeros((len(BODY_JOINTS), 3), dtype=dtype),
        qpos_joint_limits=qpos_joint_limits,
        qpos_joint_names=qpos_joint_names,
    )


def _load_xml_model_data(path: Path, *, dtype) -> SmplHumanoidWeights:
    if not path.is_file():
        raise FileNotFoundError(f"SMPL humanoid XML not found: {path}")

    root = ET.parse(path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"SMPL humanoid XML is missing a worldbody: {path}")

    parsed_bodies: dict[str, ET.Element] = {}
    parsed_parents: dict[str, str | None] = {}
    for body in worldbody.findall("body"):
        _walk_xml_bodies(body, parent_name=None, bodies=parsed_bodies, parents=parsed_parents)

    missing = sorted(set(JOINT_NAMES) - parsed_bodies.keys())
    if missing:
        raise ValueError(f"SMPL humanoid XML is missing body names: {', '.join(missing)}")

    local_offsets = np.zeros((len(JOINT_NAMES), 3), dtype=dtype)
    rest_local_rotations = np.repeat(np.eye(3, dtype=dtype)[None], len(JOINT_NAMES), axis=0)
    parsed_parent_indices = []
    by_name = {name: i for i, name in enumerate(JOINT_NAMES)}
    for joint_idx, name in enumerate(JOINT_NAMES):
        body = parsed_bodies[name]
        parent_name = parsed_parents[name]
        parsed_parent_indices.append(-1 if parent_name is None else by_name[parent_name])
        local_offsets[joint_idx] = _parse_vec(body.get("pos"), size=3, default=np.zeros(3, dtype=dtype))
        rest_local_rotations[joint_idx] = _quat_wxyz_to_matrix(
            _parse_vec(body.get("quat"), size=4, default=np.array([1.0, 0.0, 0.0, 0.0], dtype=dtype))
        )
    if parsed_parent_indices != PARENTS:
        raise ValueError("SMPL humanoid XML body hierarchy does not match the canonical SMPL hierarchy.")

    vertices, faces, link_data = _load_xml_geoms(parsed_bodies, dtype=dtype)
    return _weights_from_parts(
        local_offsets=local_offsets,
        rest_local_rotations=rest_local_rotations,
        vertices=vertices,
        faces=faces,
        link_data=link_data,
        dtype=dtype,
    )


def _walk_xml_bodies(
    body: ET.Element,
    *,
    parent_name: str | None,
    bodies: dict[str, ET.Element],
    parents: dict[str, str | None],
) -> None:
    name = body.get("name")
    if name in JOINT_NAMES:
        bodies[name] = body
        parents[name] = parent_name
        parent_name = name

    for child in body.findall("body"):
        _walk_xml_bodies(child, parent_name=parent_name, bodies=bodies, parents=parents)


def _load_xml_geoms(bodies: dict[str, ET.Element], *, dtype) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    vertices_by_link = []
    faces_by_link = []
    joint_indices = []
    vertex_starts = []
    vertex_counts = []
    face_starts = []
    face_counts = []
    geom_positions = []
    geom_rotations = []
    names = []
    vertex_offset = 0
    face_offset = 0

    for joint_idx, name in enumerate(JOINT_NAMES):
        for geom_idx, geom in enumerate(bodies[name].findall("geom")):
            vertices, faces = _geom_mesh(geom, dtype=dtype)
            vertices_by_link.append(vertices)
            faces_by_link.append(faces + vertex_offset)
            joint_indices.append(joint_idx)
            vertex_starts.append(vertex_offset)
            vertex_counts.append(vertices.shape[0])
            face_starts.append(face_offset)
            face_counts.append(faces.shape[0])
            geom_positions.append(_parse_vec(geom.get("pos"), size=3, default=np.zeros(3, dtype=dtype)))
            geom_rotations.append(
                _quat_wxyz_to_matrix(
                    _parse_vec(geom.get("quat"), size=4, default=np.array([1.0, 0.0, 0.0, 0.0], dtype=dtype))
                )
            )
            names.append(geom.get("name") or f"{name}_{geom_idx}")
            vertex_offset += vertices.shape[0]
            face_offset += faces.shape[0]

    if not vertices_by_link:
        raise ValueError("SMPL humanoid XML does not contain any primitive geoms.")

    link_data = {
        "joint_indices": joint_indices,
        "vertex_starts": vertex_starts,
        "vertex_counts": vertex_counts,
        "face_starts": face_starts,
        "face_counts": face_counts,
        "geom_positions": np.asarray(geom_positions, dtype=dtype),
        "geom_rotations": np.asarray(geom_rotations, dtype=dtype),
        "names": names,
    }
    return np.concatenate(vertices_by_link), np.concatenate(faces_by_link), link_data


def _geom_mesh(geom: ET.Element, *, dtype) -> tuple[np.ndarray, np.ndarray]:
    geom_type = geom.get("type", "sphere")
    size = _parse_vec(geom.get("size"), size=None, default=np.ones(3, dtype=dtype))
    if geom_type == "box":
        return _box(tuple(size[:3]), dtype=dtype)
    if geom_type == "sphere":
        return _sphere(float(size[0]), dtype=dtype)
    if geom_type == "capsule":
        return _capsule_between(
            np.array([0.0, 0.0, -float(size[1])], dtype=dtype),
            np.array([0.0, 0.0, float(size[1])], dtype=dtype),
            float(size[0]),
            dtype=dtype,
        )
    if geom_type == "cylinder":
        return _cylinder(float(size[0]), float(size[1]), dtype=dtype)
    raise ValueError(f"Unsupported SMPL humanoid XML geom type: {geom_type}")


def _build_link_meshes(*, dtype) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    children = {idx: [] for idx in range(len(JOINT_NAMES))}
    for joint_idx, parent in enumerate(PARENTS):
        if parent >= 0:
            children[parent].append(joint_idx)

    vertices_by_link = []
    faces_by_link = []
    joint_indices = []
    vertex_starts = []
    vertex_counts = []
    face_starts = []
    face_counts = []
    names = []
    vertex_offset = 0
    face_offset = 0

    for joint_idx, name in enumerate(JOINT_NAMES):
        child_offsets = [LOCAL_OFFSETS[child] for child in children[joint_idx]]
        if child_offsets:
            mesh_vertices = []
            mesh_faces = []
            local_offset = 0
            for child_offset in child_offsets:
                radius = _link_radius(name)
                verts, faces = _capsule_between(np.zeros(3, dtype=dtype), child_offset.astype(dtype), radius, dtype=dtype)
                mesh_vertices.append(verts)
                mesh_faces.append(faces + local_offset)
                local_offset += len(verts)
            vertices = np.concatenate(mesh_vertices, axis=0)
            faces = np.concatenate(mesh_faces, axis=0)
        else:
            vertices, faces = _box(_leaf_size(name), dtype=dtype)

        vertices_by_link.append(vertices)
        faces_by_link.append(faces + vertex_offset)
        joint_indices.append(joint_idx)
        vertex_starts.append(vertex_offset)
        vertex_counts.append(vertices.shape[0])
        face_starts.append(face_offset)
        face_counts.append(faces.shape[0])
        names.append(name)
        vertex_offset += vertices.shape[0]
        face_offset += faces.shape[0]

    link_data = {
        "joint_indices": joint_indices,
        "vertex_starts": vertex_starts,
        "vertex_counts": vertex_counts,
        "face_starts": face_starts,
        "face_counts": face_counts,
        "geom_positions": np.zeros((len(joint_indices), 3), dtype=dtype),
        "geom_rotations": np.repeat(np.eye(3, dtype=dtype)[None], len(joint_indices), axis=0),
        "names": names,
    }
    return np.concatenate(vertices_by_link), np.concatenate(faces_by_link), link_data


def _link_radius(name: str) -> float:
    if name in {"Pelvis", "Torso", "Spine", "Chest"}:
        return 0.055
    if "Hip" in name or "Knee" in name or "Ankle" in name:
        return 0.040
    if "Thorax" in name or "Shoulder" in name or "Elbow" in name:
        return 0.032
    return 0.025


def _leaf_size(name: str) -> tuple[float, float, float]:
    if "Toe" in name:
        return (0.055, 0.140, 0.035)
    if "Hand" in name:
        return (0.080, 0.035, 0.055)
    if name == "Head":
        return (0.120, 0.100, 0.130)
    return (0.050, 0.050, 0.050)


def _capsule_between(start: np.ndarray, end: np.ndarray, radius: float, *, dtype) -> tuple[np.ndarray, np.ndarray]:
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length <= 1e-8:
        return _box((radius, radius, radius), dtype=dtype)
    direction = axis / length
    basis = _basis_from_z(direction)
    sections = 10
    rings = [0.0, length]
    vertices = []
    for z in rings:
        for section in range(sections):
            angle = 2.0 * np.pi * section / sections
            local = np.array([radius * np.cos(angle), radius * np.sin(angle), z], dtype=dtype)
            vertices.append(start + basis @ local)
    vertices.append(start)
    vertices.append(end)

    faces = []
    for section in range(sections):
        nxt = (section + 1) % sections
        faces.append([section, nxt, sections + nxt])
        faces.append([section, sections + nxt, sections + section])
        faces.append([2 * sections, nxt, section])
        faces.append([sections + section, sections + nxt, 2 * sections + 1])
    return np.asarray(vertices, dtype=dtype), np.asarray(faces, dtype=np.int64)


def _basis_from_z(direction: np.ndarray) -> np.ndarray:
    z_axis = np.asarray(direction, dtype=np.float64)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-8)
    helper = np.array([0.0, 0.0, 1.0]) if abs(float(z_axis[2])) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _box(size: tuple[float, float, float], *, dtype) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = size
    vertices = np.array(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
        ],
        dtype=dtype,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _sphere(radius: float, *, dtype) -> tuple[np.ndarray, np.ndarray]:
    vertices = [np.array([0.0, 0.0, radius], dtype=dtype), np.array([0.0, 0.0, -radius], dtype=dtype)]
    rings = 4
    sections = 10
    for ring in range(1, rings):
        polar = np.pi * ring / rings
        z = radius * np.cos(polar)
        r = radius * np.sin(polar)
        for section in range(sections):
            angle = 2.0 * np.pi * section / sections
            vertices.append(np.array([r * np.cos(angle), r * np.sin(angle), z], dtype=dtype))

    faces = []
    for section in range(sections):
        nxt = (section + 1) % sections
        faces.append([0, 2 + section, 2 + nxt])
        faces.append([1, 2 + (rings - 2) * sections + nxt, 2 + (rings - 2) * sections + section])
    for ring in range(rings - 2):
        start = 2 + ring * sections
        next_start = start + sections
        for section in range(sections):
            nxt = (section + 1) % sections
            faces.append([start + section, next_start + section, next_start + nxt])
            faces.append([start + section, next_start + nxt, start + nxt])
    return np.asarray(vertices, dtype=dtype), np.asarray(faces, dtype=np.int64)


def _cylinder(radius: float, half_height: float, *, dtype) -> tuple[np.ndarray, np.ndarray]:
    sections = 12
    vertices = []
    for z in (-half_height, half_height):
        for section in range(sections):
            angle = 2.0 * np.pi * section / sections
            vertices.append(np.array([radius * np.cos(angle), radius * np.sin(angle), z], dtype=dtype))
    vertices.append(np.array([0.0, 0.0, -half_height], dtype=dtype))
    vertices.append(np.array([0.0, 0.0, half_height], dtype=dtype))

    faces = []
    for section in range(sections):
        nxt = (section + 1) % sections
        faces.append([section, nxt, sections + nxt])
        faces.append([section, sections + nxt, sections + section])
        faces.append([2 * sections, section, nxt])
        faces.append([2 * sections + 1, sections + nxt, sections + section])
    return np.asarray(vertices, dtype=dtype), np.asarray(faces, dtype=np.int64)


def _parse_vec(value: str | None, *, size: int | None, default: np.ndarray) -> np.ndarray:
    if value is None:
        return default.astype(default.dtype, copy=True)
    parsed = np.asarray([float(x) for x in value.split()], dtype=default.dtype)
    if size is not None and parsed.shape != (size,):
        raise ValueError(f"Expected vector with {size} values, got {value!r}")
    return parsed


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float32, copy=False)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=q.dtype,
    )


__all__ = [
    "ACTION_SIZE",
    "BODY_COUNT",
    "JOINT_POS_SIZE",
    "QPOS_SIZE",
    "QVEL_SIZE",
    "ROOT_QPOS_SIZE",
    "ROOT_QVEL_SIZE",
    "SmplHumanoidWeights",
    "load_model_data",
]
