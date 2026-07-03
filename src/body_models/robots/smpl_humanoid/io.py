"""MJCF loading for the rigid SMPL humanoid model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from jaxtyping import Float, Int
import trimesh.creation
from trimesh import Trimesh

from body_models import config
from body_models.cache import download_hf_archive, get_cache_dir
from body_models.robots import mjcf
from body_models.robots.smpl_humanoid.constants import BODY_JOINTS, JOINT_NAMES, PARENTS, SMPL_HUMANOID_VARIANTS

Array = Any
PathLike = Path | str
SMPL_HUMANOID_SOURCES = {name: f"{name}.xml" for name in SMPL_HUMANOID_VARIANTS}


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
    actuated_joint_indices: list[int]
    actuated_joint_limits: Float[Array, "Q 2"]
    actuated_joint_names: list[str]
    actuated_joint_types: list[str]


def load_model_data(source: PathLike = "humenv", *, dtype=np.float32) -> SmplHumanoidWeights:
    """Load a rigid SMPL humanoid from an MJCF XML file."""
    path = get_model_path(source)
    if not path.is_file():
        raise FileNotFoundError(f"SMPL humanoid XML not found: {path}")

    root = mjcf.parse_xml(path)
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
        local_offsets[joint_idx] = mjcf.parse_vec(body.get("pos"), size=3, default=np.zeros(3, dtype=dtype))
        rest_local_rotations[joint_idx] = mjcf.parse_orientation(body).astype(dtype)
    if parsed_parent_indices != PARENTS:
        raise ValueError("SMPL humanoid XML body hierarchy does not match the canonical SMPL hierarchy.")

    vertices, faces, link_data = _load_xml_geoms(parsed_bodies, dtype=dtype)
    actuated_joint_indices = [by_name[name] for name, _ in BODY_JOINTS]
    actuated_joint_names = [name for name, _ in BODY_JOINTS for _ in range(3)]
    actuated_joint_limits = _actuated_joint_limits(parsed_bodies, root=root, dtype=dtype)
    num_actuated = len(actuated_joint_names)
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
        actuated_joint_indices=actuated_joint_indices,
        actuated_joint_limits=actuated_joint_limits,
        actuated_joint_names=actuated_joint_names,
        actuated_joint_types=["hinge"] * num_actuated,
    )


def get_model_path(source: PathLike = "humenv") -> Path:
    """Resolve a SMPL humanoid XML file, downloading named sources when needed."""
    if isinstance(source, str):
        name = source.strip().lower().replace("-", "_")
        if name in SMPL_HUMANOID_SOURCES:
            model_path = config.get_model_path(f"smpl-humanoid-{name}")
            return validate_path(model_path) if model_path is not None else download_model(name)
        path = Path(source)
        if path.is_file():
            return path
        if not path.parent.parts:
            variants = ", ".join(SMPL_HUMANOID_VARIANTS)
            raise ValueError(f"Unknown SMPL humanoid source {source!r}. Available sources: {variants}")

    return Path(source)


def download_model(source: str = "humenv") -> Path:
    name = source.strip().lower().replace("-", "_")
    if name not in SMPL_HUMANOID_SOURCES:
        variants = ", ".join(SMPL_HUMANOID_VARIANTS)
        raise ValueError(f"Unknown SMPL humanoid source {source!r}. Available sources: {variants}")
    cache_dir = get_cache_dir() / "smpl_humanoid"
    path = cache_dir / SMPL_HUMANOID_SOURCES[name]
    if not path.is_file():
        download_hf_archive("smpl_humanoid/assets.zip", cache_dir)
    return path


def validate_path(path: PathLike) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".xml":
        raise ValueError(f"Expected a SMPL humanoid XML file, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"SMPL humanoid XML not found: {path}")
    return path


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


def _actuated_joint_limits(
    bodies: dict[str, ET.Element],
    *,
    root: ET.Element,
    dtype,
) -> Float[np.ndarray, "Q 2"]:
    compiler = root.find("compiler")
    angle_scale = 1.0 if compiler is not None and compiler.get("angle") == "radian" else np.pi / 180.0
    limits = []
    for joint_name, _ in BODY_JOINTS:
        joints = {joint.get("name"): joint for joint in bodies[joint_name].findall("joint")}
        for axis in ("x", "y", "z"):
            joint = joints.get(f"{joint_name}_{axis}")
            if joint is None:
                limits.append([-np.pi, np.pi])
                continue
            lo, hi = (float(value) for value in joint.attrib["range"].split())
            limits.append([angle_scale * lo, angle_scale * hi])
    return np.asarray(limits, dtype=dtype)


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
            geom_position, geom_rotation = _geom_transform(geom, dtype=dtype)
            vertices_by_link.append(vertices)
            faces_by_link.append(faces + vertex_offset)
            joint_indices.append(joint_idx)
            vertex_starts.append(vertex_offset)
            vertex_counts.append(vertices.shape[0])
            face_starts.append(face_offset)
            face_counts.append(faces.shape[0])
            geom_positions.append(geom_position)
            geom_rotations.append(geom_rotation)
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
    size = mjcf.parse_vec(geom.get("size"), size=None, default=np.ones(3, dtype=dtype))
    if geom_type == "box":
        return _mesh_arrays(trimesh.creation.box(extents=2.0 * size[:3]), dtype=dtype)
    if geom_type == "sphere":
        return _mesh_arrays(trimesh.creation.uv_sphere(radius=float(size[0]), count=(8, 16)), dtype=dtype)
    if geom_type == "capsule":
        fromto = geom.get("fromto")
        if fromto is not None:
            capsule = mjcf.parse_vec(fromto, size=6, default=np.zeros(6, dtype=dtype))
            height = float(np.linalg.norm(capsule[3:] - capsule[:3]))
        else:
            height = 2.0 * float(size[1])
        return _mesh_arrays(trimesh.creation.capsule(height=height, radius=float(size[0]), count=(8, 16)), dtype=dtype)
    if geom_type == "cylinder":
        return _mesh_arrays(trimesh.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1])), dtype=dtype)
    raise ValueError(f"Unsupported SMPL humanoid XML geom type: {geom_type}")


def _geom_transform(geom: ET.Element, *, dtype) -> tuple[np.ndarray, np.ndarray]:
    fromto = geom.get("fromto")
    if fromto is None:
        position = mjcf.parse_vec(geom.get("pos"), size=3, default=np.zeros(3, dtype=dtype))
        rotation = mjcf.parse_orientation(geom).astype(dtype)
        return position, rotation

    capsule = mjcf.parse_vec(fromto, size=6, default=np.zeros(6, dtype=dtype))
    start = capsule[:3]
    end = capsule[3:]
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length <= 1e-8:
        raise ValueError("Capsule endpoints must be distinct.")
    return 0.5 * (start + end), _basis_from_z(axis / length).astype(dtype)


def _basis_from_z(direction: np.ndarray) -> np.ndarray:
    z_axis = np.asarray(direction, dtype=np.float64)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-8)
    helper = np.array([0.0, 0.0, 1.0]) if abs(float(z_axis[2])) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _mesh_arrays(mesh: Trimesh, *, dtype) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(mesh.vertices, dtype=dtype), np.asarray(mesh.faces, dtype=np.int64)


__all__ = [
    "SMPL_HUMANOID_SOURCES",
    "SmplHumanoidWeights",
    "download_model",
    "get_model_path",
    "load_model_data",
    "validate_path",
]
