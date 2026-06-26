"""I/O utilities for the BrainCo Revo 2 robotic hand model."""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int

from body_models import config
from body_models.cache import get_cache_dir
from body_models.common.stl import load_stl_mesh as _load_stl_mesh
from body_models.robots import mjcf

PathLike = Path | str
Side = Literal["left", "right"]
Array = Any
VALID_SIDES = ("left", "right")
BRAINCO_REVO2_MUJOCO_URL = (
    "https://brainco-common-public.oss-cn-hangzhou.aliyuncs.com/web-config/docs-sdk/Revo2_xml.zip"
)
MUJOCO_TO_KIMODO = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
JOINT_SUFFIXES = [
    "base_skel",
    "thumb_metacarpal_skel",
    "thumb_proximal_skel",
    "thumb_distal_skel",
    "index_proximal_skel",
    "index_distal_skel",
    "middle_proximal_skel",
    "middle_distal_skel",
    "ring_proximal_skel",
    "ring_distal_skel",
    "pinky_proximal_skel",
    "pinky_distal_skel",
]
PARENTS = [-1, 0, 1, 2, 0, 4, 0, 6, 0, 8, 0, 10]
ACTIVE_JOINT_SUFFIXES = {
    "thumb_metacarpal_skel",
    "thumb_proximal_skel",
    "index_proximal_skel",
    "middle_proximal_skel",
    "ring_proximal_skel",
    "pinky_proximal_skel",
}


@dataclass(frozen=True)
class BrainCoWeights:
    side: Side
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
    actuated_joint_axes: Float[Array, "Q 3"]
    actuated_joint_limits: Float[Array, "Q 2"]
    actuated_joint_names: list[str]
    coupled_joint_indices: list[int]
    coupled_joint_axes: Float[Array, "C 3"]
    coupled_driver_indices: list[int]
    coupled_polycoef: Float[Array, "C 4"]


def get_model_path(model_path: PathLike | None = None) -> Path:
    """Resolve a BrainCo asset directory containing official Revo 2 MuJoCo XML and STL files."""
    if model_path is None:
        model_path = config.get_model_path("brainco")
    if model_path is not None:
        return validate_path(model_path)
    cache_path = get_cache_dir() / "brainco"
    if _has_model(cache_path):
        return cache_path
    return download_model()


def download_model() -> Path:
    """Download the official BrainCo Revo 2 MuJoCo model package."""
    cache_dir = get_cache_dir() / "brainco"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    archive_path = cache_dir / "revo2_mujoco.zip"
    urllib.request.urlretrieve(BRAINCO_REVO2_MUJOCO_URL, archive_path)
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            side = _archive_side(member.filename)
            if side is None:
                continue
            name = Path(member.filename).name
            if name.endswith(".xml") and name.startswith("brainco-"):
                target = cache_dir / f"{side}.xml"
            elif "/meshes/" in member.filename and name.endswith(".STL"):
                target = cache_dir / "meshes" / side / _mesh_name(side, name)
            else:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    archive_path.unlink(missing_ok=True)
    return validate_path(cache_dir)


def validate_path(path: PathLike) -> Path:
    path = Path(path)
    if path.is_file():
        raise ValueError(f"Expected a BrainCo asset directory, got file: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"BrainCo model directory not found: {path}")
    for side in VALID_SIDES:
        if not (path / f"{side}.xml").exists():
            raise FileNotFoundError(f"BrainCo XML not found: {path / f'{side}.xml'}")
        if not (path / "meshes" / side).is_dir():
            raise FileNotFoundError(f"BrainCo mesh directory not found: {path / 'meshes' / side}")
    return path


def load_model_data(model_path: PathLike | None = None, *, side: Side = "right", dtype=np.float32) -> BrainCoWeights:
    if side not in VALID_SIDES:
        raise ValueError(f"Invalid BrainCo side: {side}")
    model_dir = get_model_path(model_path)
    root = mjcf.parse_xml(model_dir / f"{side}.xml")
    names = [f"{side}_{suffix}" for suffix in JOINT_SUFFIXES]
    class_axes, class_limits = mjcf.joint_defaults(root)
    local_offsets, rest_local_rotations, mesh_transforms = _parse_rest_and_mesh_transforms(root, side, names)
    joint_indices, joint_axes, joint_limits, joint_names = _parse_active_joints(root, names, class_axes, class_limits)
    coupled_joint_indices, coupled_joint_axes, coupled_driver_indices, coupled_polycoef = _parse_coupled_joints(
        root,
        names,
        joint_names,
        class_axes,
    )
    vertices, faces, link_data = _load_link_meshes(model_dir / "meshes" / side, mesh_transforms, names, dtype=dtype)
    return BrainCoWeights(
        side=side,
        joint_names=names,
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
        actuated_joint_indices=joint_indices,
        actuated_joint_axes=joint_axes.astype(dtype),
        actuated_joint_limits=joint_limits.astype(dtype),
        actuated_joint_names=joint_names,
        coupled_joint_indices=coupled_joint_indices,
        coupled_joint_axes=coupled_joint_axes.astype(dtype),
        coupled_driver_indices=coupled_driver_indices,
        coupled_polycoef=coupled_polycoef.astype(dtype),
    )


def _parse_rest_and_mesh_transforms(
    root: ET.Element,
    side: str,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    offsets = np.zeros((len(names), 3), dtype=np.float32)
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], len(names), axis=0)
    mesh_transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    mesh_file_by_name = mjcf.mesh_files_by_name(root)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("BrainCo XML is missing a worldbody")
    base = worldbody.find("body")
    if base is None:
        raise ValueError("BrainCo XML is missing a root hand body")

    by_name = {name: i for i, name in enumerate(names)}
    base_rot = mjcf.parse_orientation(base)
    rotations[0] = MUJOCO_TO_KIMODO @ base_rot @ MUJOCO_TO_KIMODO.T
    _add_mesh_transforms(base, side, mesh_file_by_name, base_rot, mesh_transforms)

    def walk(body: ET.Element, parent_pos: np.ndarray, parent_rot: np.ndarray, fold_parent: bool) -> None:
        body_pos = mjcf.parse_vec(body.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
        body_rot = mjcf.parse_orientation(body)
        local_pos = parent_pos + parent_rot @ body_pos if fold_parent else body_pos
        local_rot = parent_rot @ body_rot if fold_parent else body_rot
        name = _side_name(side, _body_to_joint_name(body))
        if name in by_name:
            offsets[by_name[name]] = MUJOCO_TO_KIMODO @ local_pos
            rotations[by_name[name]] = MUJOCO_TO_KIMODO @ local_rot @ MUJOCO_TO_KIMODO.T
        _add_mesh_transforms(body, side, mesh_file_by_name, np.eye(3, dtype=np.float32), mesh_transforms)
        for child in body.findall("body"):
            walk(child, np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32), False)

    for child in base.findall("body"):
        walk(child, np.zeros(3, dtype=np.float32), base_rot, True)
    return offsets, rotations, mesh_transforms


def _parse_active_joints(
    root: ET.Element,
    names: list[str],
    class_axes: dict[str, np.ndarray],
    class_limits: dict[str, tuple[float, float]],
) -> tuple[list[int], np.ndarray, np.ndarray, list[str]]:
    by_name = {name: i for i, name in enumerate(names)}
    indices = []
    axes = []
    limits = []
    joint_names = []
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if not name:
            continue
        skel_name = name.replace("_joint", "_skel")
        suffix = skel_name.split("_", 1)[1]
        if suffix not in ACTIVE_JOINT_SUFFIXES:
            continue
        axis = MUJOCO_TO_KIMODO @ _joint_axis(joint, class_axes)
        indices.append(by_name[skel_name])
        axes.append(axis / np.linalg.norm(axis))
        limits.append(_joint_limit(joint, class_limits))
        joint_names.append(skel_name)
    return indices, np.asarray(axes), np.asarray(limits), joint_names


def _parse_coupled_joints(
    root: ET.Element,
    names: list[str],
    actuated_joint_names: list[str],
    class_axes: dict[str, np.ndarray],
) -> tuple[list[int], np.ndarray, list[int], np.ndarray]:
    by_name = {name: i for i, name in enumerate(names)}
    qpos_by_name = {name: i for i, name in enumerate(actuated_joint_names)}
    joint_by_name = {joint.get("name"): joint for joint in root.findall(".//joint") if joint.get("name")}
    indices = []
    axes = []
    drivers = []
    polycoefs = []
    for equality in root.findall(".//equality/joint"):
        joint1 = equality.get("joint1")
        joint2 = equality.get("joint2")
        if joint1 is None or joint2 is None:
            continue
        driver_name = joint1.replace("_joint", "_skel")
        coupled_name = joint2.replace("_joint", "_skel")
        if driver_name not in qpos_by_name or coupled_name not in by_name:
            continue
        coupled_joint = joint_by_name[joint2]
        axis = MUJOCO_TO_KIMODO @ _joint_axis(coupled_joint, class_axes)
        polycoef = mjcf.parse_vec(equality.get("polycoef"), default=np.array([0, 1, 0, 0], dtype=np.float32))
        if polycoef.shape != (4,):
            raise ValueError(f"BrainCo equality joint {joint2} must have four polycoef values")
        indices.append(by_name[coupled_name])
        axes.append(axis / np.linalg.norm(axis))
        drivers.append(qpos_by_name[driver_name])
        polycoefs.append(polycoef)
    return indices, np.asarray(axes), drivers, np.asarray(polycoefs)


def _load_link_meshes(
    mesh_dir: Path,
    mesh_transforms: dict[str, tuple[np.ndarray, np.ndarray]],
    joint_names: list[str],
    *,
    dtype,
) -> tuple[np.ndarray, np.ndarray, dict]:
    vertices_by_link = []
    faces_by_link = []
    link_data = {
        "joint_indices": [],
        "vertex_starts": [],
        "vertex_counts": [],
        "face_starts": [],
        "face_counts": [],
        "geom_positions": [],
        "geom_rotations": [],
        "names": [],
    }
    vertex_offset = 0
    face_offset = 0
    by_name = {name: i for i, name in enumerate(joint_names)}
    for mesh_file, (geom_pos, geom_rot) in mesh_transforms.items():
        path = mesh_dir / mesh_file
        if not path.exists():
            raise FileNotFoundError(f"BrainCo mesh not found: {path}")
        vertices, faces = load_stl_mesh(path, dtype=dtype)
        vertices_by_link.append(vertices)
        faces_by_link.append(faces + vertex_offset)
        link_data["joint_indices"].append(by_name[_mesh_joint_name(mesh_file)])
        link_data["vertex_starts"].append(vertex_offset)
        link_data["vertex_counts"].append(vertices.shape[0])
        link_data["face_starts"].append(face_offset)
        link_data["face_counts"].append(faces.shape[0])
        link_data["geom_positions"].append(geom_pos)
        link_data["geom_rotations"].append(geom_rot)
        link_data["names"].append(mesh_file)
        vertex_offset += vertices.shape[0]
        face_offset += faces.shape[0]
    if not vertices_by_link:
        raise FileNotFoundError(f"No BrainCo STL link meshes found in {mesh_dir}")
    link_data["geom_positions"] = np.asarray(link_data["geom_positions"])
    link_data["geom_rotations"] = np.asarray(link_data["geom_rotations"])
    return np.concatenate(vertices_by_link), np.concatenate(faces_by_link), link_data


def load_stl_mesh(path: Path, *, dtype=np.float32) -> tuple[Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"]]:
    return _load_stl_mesh(path, coord=MUJOCO_TO_KIMODO, dtype=dtype)


def _add_mesh_transforms(
    body: ET.Element,
    side: str,
    mesh_file_by_name: dict[str, str],
    base_rot: np.ndarray,
    out: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    for geom in body.findall("geom"):
        mesh_name = geom.get("mesh")
        if mesh_name is None:
            continue
        mesh_file = mesh_file_by_name.get(mesh_name)
        if mesh_file is None:
            raise FileNotFoundError(f"BrainCo XML references missing mesh asset: {mesh_name}")
        name = _mesh_name(side, Path(mesh_file).name)
        if name in out:
            continue
        pos = mjcf.parse_vec(geom.get("pos"), default=np.zeros(3, dtype=np.float32), size=3)
        rot = mjcf.parse_orientation(geom)
        out[name] = (MUJOCO_TO_KIMODO @ (base_rot @ pos), MUJOCO_TO_KIMODO @ (base_rot @ rot) @ MUJOCO_TO_KIMODO.T)


def _body_to_joint_name(body: ET.Element) -> str:
    joint = body.find("joint")
    joint_name = joint.get("name") if joint is not None else None
    if joint_name:
        return joint_name.replace("_joint", "_skel")
    return body.get("name", "").removesuffix("_link") + "_skel"


def _side_name(side: str, name: str) -> str:
    return name if name.startswith(f"{side}_") else f"{side}_{name}"


def _mesh_joint_name(mesh_file: str) -> str:
    stem = mesh_file.removesuffix(".STL")
    side, suffix = stem.split("_", 1)
    if suffix == "base_link":
        return f"{side}_base_skel"
    if suffix == "base_visual_link":
        return f"{side}_base_skel"
    if suffix.endswith("_touch_link"):
        return f"{side}_{suffix.removesuffix('_touch_link')}_distal_skel"
    if suffix.endswith("_tip"):
        return f"{side}_{suffix.removesuffix('_tip')}_distal_skel"
    if suffix.endswith("_tip_link"):
        return f"{side}_{suffix.removesuffix('_tip_link')}_distal_skel"
    if suffix == "thumb_proximal_visual_link":
        return f"{side}_thumb_proximal_skel"
    return f"{side}_{suffix.removesuffix('_link')}_skel"


def _joint_axis(joint: ET.Element, class_axes: dict[str, np.ndarray]) -> np.ndarray:
    if joint.get("axis"):
        return mjcf.parse_vec(joint.get("axis"), default=np.zeros(3, dtype=np.float32), size=3)
    class_name = joint.get("class")
    if class_name in class_axes:
        return class_axes[class_name]
    raise ValueError(f"Missing axis for BrainCo joint {joint.get('name')}")


def _joint_limit(joint: ET.Element, class_limits: dict[str, tuple[float, float]]) -> tuple[float, float]:
    limit = joint.get("range")
    if limit is not None:
        lo, hi = [float(x) for x in limit.split()]
        return lo, hi
    class_name = joint.get("class")
    if class_name in class_limits:
        return class_limits[class_name]
    raise ValueError(f"Missing limit for BrainCo joint {joint.get('name')}")


def _archive_side(filename: str) -> str | None:
    if "/xml_left/" in filename:
        return "left"
    if "/xml_right/" in filename:
        return "right"
    return None


def _mesh_name(side: str, name: str) -> str:
    return name if name.startswith(f"{side}_") else f"{side}_{name}"


def _has_model(path: Path) -> bool:
    return all((path / f"{side}.xml").exists() and (path / "meshes" / side).is_dir() for side in VALID_SIDES)
