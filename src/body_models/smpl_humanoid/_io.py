"""MJCF loading for the rigid SMPL humanoid model."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh.creation
from jaxtyping import Float, Int
from trimesh import Trimesh

from body_models import _config as config
from body_models._cache import download_hf_archive, get_cache_dir
from body_models._common import mjcf
from body_models.smpl_humanoid._constants import (
    BODY_JOINTS,
    FINGER_JOINT_NAMES,
    JOINT_NAMES,
    PARENTS,
    ROBOT_JOINT_NAMES,
    ROBOT_PARENTS,
    SMPL_HUMANOID_VARIANTS,
)

Array = Any
PathLike = Path | str
SMPL_HUMANOID_SOURCES = {
    "mannequin": "mannequin_lods/lod0_40k/neutral.xml",
    "mannequin_lod1": "mannequin_lods/lod1_15k/neutral.xml",
    "mannequin_lod2": "mannequin_lods/lod2_5k/neutral.xml",
    "humenv": "humenv.xml",
    "phc": "phc.xml",
    "smplsim": "smplsim.xml",
}


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
    present_fingers = set(FINGER_JOINT_NAMES) & parsed_bodies.keys()
    if present_fingers and present_fingers != set(FINGER_JOINT_NAMES):
        missing_fingers = sorted(set(FINGER_JOINT_NAMES) - present_fingers)
        raise ValueError(f"SMPL humanoid XML has an incomplete finger hierarchy: {', '.join(missing_fingers)}")
    has_fingers = bool(present_fingers)
    joint_names = ROBOT_JOINT_NAMES if has_fingers else JOINT_NAMES
    expected_parents = ROBOT_PARENTS if has_fingers else PARENTS

    local_offsets = np.zeros((len(joint_names), 3), dtype=dtype)
    rest_local_rotations = np.repeat(np.eye(3, dtype=dtype)[None], len(joint_names), axis=0)
    parsed_parent_indices = []
    by_name = {name: i for i, name in enumerate(joint_names)}
    for joint_idx, name in enumerate(joint_names):
        body = parsed_bodies[name]
        parent_name = parsed_parents[name]
        parsed_parent_indices.append(-1 if parent_name is None else by_name[parent_name])
        local_offsets[joint_idx] = mjcf.parse_vec(body.get("pos"), size=3, default=np.zeros(3, dtype=dtype))
        rest_local_rotations[joint_idx] = mjcf.parse_orientation(body).astype(dtype)
    if parsed_parent_indices != expected_parents:
        raise ValueError("SMPL humanoid XML body hierarchy does not match the canonical SMPL hierarchy.")

    vertices, faces, link_data = _load_xml_geoms(
        parsed_bodies,
        joint_names=joint_names,
        root=root,
        base_dir=path.parent,
        dtype=dtype,
    )
    actuated_names = [name for name, _ in BODY_JOINTS]
    if has_fingers:
        actuated_names.extend(FINGER_JOINT_NAMES)
    actuated_joint_indices = [by_name[name] for name in actuated_names]
    actuated_joint_names = [name for name in actuated_names for _ in range(3)]
    actuated_joint_limits = _actuated_joint_limits(parsed_bodies, actuated_names=actuated_names, root=root, dtype=dtype)
    num_dofs = len(actuated_joint_names)
    return SmplHumanoidWeights(
        joint_names=list(joint_names),
        parents=list(expected_parents),
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
        actuated_joint_types=["hinge"] * num_dofs,
    )


def get_model_path(source: PathLike = "humenv") -> Path:
    """Resolve a SMPL humanoid XML file, downloading named sources when needed."""
    if isinstance(source, str):
        name = source.strip().lower().replace("-", "_")
        if name in SMPL_HUMANOID_SOURCES:
            model_path = config.get_model_path(f"smpl-humanoid-{name.replace('_', '-')}")
            return validate_path(model_path) if model_path is not None else download_model(name)
        path = Path(source)
        if path.is_file():
            return path
        if not path.parent.parts:
            variants = ", ".join(SMPL_HUMANOID_VARIANTS)
            raise ValueError(f"Unknown SMPL humanoid source {source!r}. Available sources: {variants}")

    return Path(source)


def download_model(source: str = "humenv", output_dir: PathLike | None = None) -> Path:
    name = source.strip().lower().replace("-", "_")
    if name not in SMPL_HUMANOID_SOURCES:
        variants = ", ".join(SMPL_HUMANOID_VARIANTS)
        raise ValueError(f"Unknown SMPL humanoid source {source!r}. Available sources: {variants}")
    output_dir = Path(output_dir) if output_dir is not None else get_cache_dir() / "smpl_humanoid"
    path = output_dir / SMPL_HUMANOID_SOURCES[name]
    if not path.is_file():
        download_hf_archive("smpl_humanoid/assets.zip", output_dir)
    return validate_path(path)


def download_assets(output_dir: PathLike | None = None) -> dict[str, Path]:
    """Download every configured SMPL humanoid variant."""
    return {
        f"smpl-humanoid-{source}": download_model(source, output_dir=output_dir) for source in SMPL_HUMANOID_VARIANTS
    }


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
    if name in ROBOT_JOINT_NAMES:
        bodies[name] = body
        parents[name] = parent_name
        parent_name = name

    for child in body.findall("body"):
        _walk_xml_bodies(child, parent_name=parent_name, bodies=bodies, parents=parents)


def _actuated_joint_limits(
    bodies: dict[str, ET.Element],
    *,
    actuated_names: list[str],
    root: ET.Element,
    dtype,
) -> Float[np.ndarray, "Q 2"]:
    compiler = root.find("compiler")
    angle_scale = 1.0 if compiler is not None and compiler.get("angle") == "radian" else np.pi / 180.0
    limits = []
    for joint_name in actuated_names:
        joints = {joint.get("name"): joint for joint in bodies[joint_name].findall("joint")}
        for axis in ("x", "y", "z"):
            joint = joints.get(f"{joint_name}_{axis}")
            if joint is None:
                limits.append([-np.pi, np.pi])
                continue
            lo, hi = (float(value) for value in joint.attrib["range"].split())
            limits.append([angle_scale * lo, angle_scale * hi])
    return np.asarray(limits, dtype=dtype)


def _load_xml_geoms(
    bodies: dict[str, ET.Element],
    *,
    joint_names: list[str] | tuple[str, ...],
    root: ET.Element,
    base_dir: Path,
    dtype,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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

    mesh_assets = _mesh_assets(root, base_dir=base_dir, dtype=dtype)
    for joint_idx, name in enumerate(joint_names):
        for geom_idx, geom in enumerate(bodies[name].findall("geom")):
            vertices, faces = _geom_mesh(geom, mesh_assets=mesh_assets, dtype=dtype)
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


def _mesh_assets(root: ET.Element, *, base_dir: Path, dtype) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    assets = {}
    asset_element = root.find("asset")
    if asset_element is None:
        return assets
    for mesh_element in asset_element.findall("mesh"):
        name = mesh_element.get("name")
        file = mesh_element.get("file")
        if not name or not file:
            raise ValueError("MJCF mesh assets require both name and file attributes.")
        mesh = trimesh.load_mesh(base_dir / file, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        if not isinstance(mesh, Trimesh):
            raise TypeError(f"MJCF mesh asset is not triangular: {file}")
        vertices, faces = _mesh_arrays(mesh, dtype=dtype)
        scale = mjcf.parse_vec(mesh_element.get("scale"), size=3, default=np.ones(3, dtype=dtype))
        assets[name] = (vertices * scale, faces)
    return assets


def _geom_mesh(
    geom: ET.Element,
    *,
    mesh_assets: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    dtype,
) -> tuple[np.ndarray, np.ndarray]:
    geom_type = geom.get("type", "sphere")
    if geom_type == "mesh":
        mesh_name = geom.get("mesh")
        if not mesh_name or mesh_assets is None or mesh_name not in mesh_assets:
            raise ValueError(f"Unknown MJCF mesh asset: {mesh_name!r}")
        return mesh_assets[mesh_name]
    size = mjcf.parse_vec(geom.get("size"), size=None, default=np.ones(3, dtype=dtype))
    if geom_type == "box":
        return _mesh_arrays(trimesh.creation.box(extents=2.0 * size[:3]), dtype=dtype)
    if geom_type == "sphere":
        return _mesh_arrays(trimesh.creation.uv_sphere(radius=float(size[0]), count=(8, 16)), dtype=dtype)
    if geom_type == "ellipsoid":
        mesh = trimesh.creation.uv_sphere(radius=1.0, count=(12, 24))
        mesh.vertices *= size[:3]
        return _mesh_arrays(mesh, dtype=dtype)
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


def _geom_transform(
    geom: ET.Element,
    *,
    dtype,
) -> tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3 3"]]:
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


def _basis_from_z(direction: Float[np.ndarray, "3"]) -> Float[np.ndarray, "3 3"]:
    z_axis = np.asarray(direction, dtype=np.float64)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-8)
    helper = np.array([0.0, 0.0, 1.0]) if abs(float(z_axis[2])) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _mesh_arrays(
    mesh: Trimesh,
    *,
    dtype,
) -> tuple[Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"]]:
    return np.asarray(mesh.vertices, dtype=dtype), np.asarray(mesh.faces, dtype=np.int64)


__all__ = [
    "SMPL_HUMANOID_SOURCES",
    "SmplHumanoidWeights",
    "download_model",
    "get_model_path",
    "load_model_data",
    "validate_path",
]
