"""I/O utilities for the Unitree G1 rigid model."""

from __future__ import annotations

import struct
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .. import config
from ..utils import get_cache_dir

PathLike = Path | str

MUJOCO_TO_KIMODO = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
G1_HF_BASE_URL = "https://huggingface.co/lerobot/unitree-g1-mujoco/resolve/main/assets"
G1_HF_XML = "g1_29dof_no_hand.xml"

JOINT_NAMES = [
    "pelvis_skel",
    "left_hip_pitch_skel",
    "left_hip_roll_skel",
    "left_hip_yaw_skel",
    "left_knee_skel",
    "left_ankle_pitch_skel",
    "left_ankle_roll_skel",
    "left_toe_base",
    "right_hip_pitch_skel",
    "right_hip_roll_skel",
    "right_hip_yaw_skel",
    "right_knee_skel",
    "right_ankle_pitch_skel",
    "right_ankle_roll_skel",
    "right_toe_base",
    "waist_yaw_skel",
    "waist_roll_skel",
    "waist_pitch_skel",
    "left_shoulder_pitch_skel",
    "left_shoulder_roll_skel",
    "left_shoulder_yaw_skel",
    "left_elbow_skel",
    "left_wrist_roll_skel",
    "left_wrist_pitch_skel",
    "left_wrist_yaw_skel",
    "left_hand_roll_skel",
    "right_shoulder_pitch_skel",
    "right_shoulder_roll_skel",
    "right_shoulder_yaw_skel",
    "right_elbow_skel",
    "right_wrist_roll_skel",
    "right_wrist_pitch_skel",
    "right_wrist_yaw_skel",
    "right_hand_roll_skel",
]

PARENTS = [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    0,
    8,
    9,
    10,
    11,
    12,
    13,
    0,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    17,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
]

G1_MESH_JOINT_MAP = {
    "pelvis_skel": ["pelvis.STL", "pelvis_contour_link.STL"],
    "left_hip_pitch_skel": ["left_hip_pitch_link.STL"],
    "left_hip_roll_skel": ["left_hip_roll_link.STL"],
    "left_hip_yaw_skel": ["left_hip_yaw_link.STL"],
    "left_knee_skel": ["left_knee_link.STL"],
    "left_ankle_pitch_skel": ["left_ankle_pitch_link.STL"],
    "left_ankle_roll_skel": ["left_ankle_roll_link.STL"],
    "right_hip_pitch_skel": ["right_hip_pitch_link.STL"],
    "right_hip_roll_skel": ["right_hip_roll_link.STL"],
    "right_hip_yaw_skel": ["right_hip_yaw_link.STL"],
    "right_knee_skel": ["right_knee_link.STL"],
    "right_ankle_pitch_skel": ["right_ankle_pitch_link.STL"],
    "right_ankle_roll_skel": ["right_ankle_roll_link.STL"],
    "waist_yaw_skel": ["waist_yaw_link.STL"],
    "waist_roll_skel": ["waist_roll_link.STL"],
    "waist_pitch_skel": ["torso_link.STL", "logo_link.STL", "head_link.STL"],
    "left_shoulder_pitch_skel": ["left_shoulder_pitch_link.STL"],
    "left_shoulder_roll_skel": ["left_shoulder_roll_link.STL"],
    "left_shoulder_yaw_skel": ["left_shoulder_yaw_link.STL"],
    "left_elbow_skel": ["left_elbow_link.STL"],
    "left_wrist_roll_skel": ["left_wrist_roll_link.STL"],
    "left_wrist_pitch_skel": ["left_wrist_pitch_link.STL"],
    "left_wrist_yaw_skel": ["left_wrist_yaw_link.STL", "left_rubber_hand.STL"],
    "right_shoulder_pitch_skel": ["right_shoulder_pitch_link.STL"],
    "right_shoulder_roll_skel": ["right_shoulder_roll_link.STL"],
    "right_shoulder_yaw_skel": ["right_shoulder_yaw_link.STL"],
    "right_elbow_skel": ["right_elbow_link.STL"],
    "right_wrist_roll_skel": ["right_wrist_roll_link.STL"],
    "right_wrist_pitch_skel": ["right_wrist_pitch_link.STL"],
    "right_wrist_yaw_skel": ["right_wrist_yaw_link.STL", "right_rubber_hand.STL"],
}


def get_model_path(model_path: PathLike | None = None) -> Path:
    """Resolve a G1 asset directory containing ``xml/g1.xml`` and ``meshes/g1``."""
    if model_path is None:
        model_path = config.get_model_path("g1")

    if model_path is not None:
        return validate_path(model_path)

    cache_path = get_cache_dir() / "g1"
    if (cache_path / "xml" / "g1.xml").exists() and (cache_path / "meshes" / "g1").is_dir():
        return cache_path

    return download_model()


def download_model() -> Path:
    """Download G1 XML and STL assets from Hugging Face."""
    cache_dir = get_cache_dir() / "g1"
    xml_dir = cache_dir / "xml"
    mesh_dir = cache_dir / "meshes" / "g1"
    xml_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading G1 model to {cache_dir}...")
    urllib.request.urlretrieve(f"{G1_HF_BASE_URL}/{G1_HF_XML}", xml_dir / "g1.xml")
    for mesh_name in sorted({mesh for meshes in G1_MESH_JOINT_MAP.values() for mesh in meshes}):
        urllib.request.urlretrieve(f"{G1_HF_BASE_URL}/meshes/{mesh_name}", mesh_dir / mesh_name)
    print("Done")
    return cache_dir


def validate_path(path: PathLike) -> Path:
    path = Path(path)
    if path.is_file():
        raise ValueError(f"Expected a G1 asset directory, got file: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"G1 model directory not found: {path}")
    xml_path = path / "xml" / "g1.xml"
    mesh_dir = path / "meshes" / "g1"
    if not xml_path.exists():
        raise FileNotFoundError(f"G1 XML not found: {xml_path}")
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"G1 mesh directory not found: {mesh_dir}")
    return path


def load_model_data(model_path: PathLike | None = None, *, dtype=np.float32) -> dict:
    model_dir = get_model_path(model_path)
    xml_path = model_dir / "xml" / "g1.xml"
    mesh_dir = model_dir / "meshes" / "g1"
    tree = ET.parse(xml_path)
    root = tree.getroot()

    class_axes, class_limits = _parse_joint_defaults(root)
    local_offsets, rest_local_rotations = _parse_joint_rest(root)
    mesh_transforms = _parse_mesh_local_transforms(root)
    qpos_joint_indices, qpos_joint_axes, qpos_joint_limits, qpos_joint_names = _parse_qpos_joints(
        root,
        class_axes,
        class_limits,
    )
    vertices, faces, link_data = _load_link_meshes(mesh_dir, mesh_transforms, dtype=dtype)
    return {
        "joint_names": JOINT_NAMES.copy(),
        "parents": PARENTS.copy(),
        "local_offsets": local_offsets.astype(dtype),
        "rest_local_rotations": rest_local_rotations.astype(dtype),
        "vertices": vertices.astype(dtype),
        "faces": faces.astype(np.int64),
        "link_joint_indices": link_data["joint_indices"],
        "link_vertex_starts": link_data["vertex_starts"],
        "link_vertex_counts": link_data["vertex_counts"],
        "link_face_starts": link_data["face_starts"],
        "link_face_counts": link_data["face_counts"],
        "link_geom_positions": link_data["geom_positions"].astype(dtype),
        "link_geom_rotations": link_data["geom_rotations"].astype(dtype),
        "link_names": link_data["names"],
        "qpos_joint_indices": qpos_joint_indices,
        "qpos_joint_axes": qpos_joint_axes.astype(dtype),
        "qpos_joint_limits": qpos_joint_limits.astype(dtype),
        "qpos_joint_names": qpos_joint_names,
    }


def _parse_joint_defaults(root: ET.Element) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, float]]]:
    class_axes: dict[str, np.ndarray] = {}
    class_limits: dict[str, tuple[float, float]] = {}
    for xml_class in root.findall(".//default"):
        class_name = xml_class.get("class")
        if not class_name:
            continue
        joint = xml_class.find("joint")
        if joint is None:
            continue
        axis = joint.get("axis")
        if axis:
            class_axes[class_name] = np.array([float(x) for x in axis.split()], dtype=np.float32)
        limit = joint.get("range")
        if limit:
            lo, hi = [float(x) for x in limit.split()]
            class_limits[class_name] = (lo, hi)
    return class_axes, class_limits


def _parse_joint_rest(root: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    local_offsets = np.zeros((len(JOINT_NAMES), 3), dtype=np.float32)
    rest_local_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], len(JOINT_NAMES), axis=0)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("g1.xml is missing a worldbody")

    by_name = {name: i for i, name in enumerate(JOINT_NAMES)}

    def walk(body: ET.Element) -> None:
        joint_name = _body_to_joint_name(body)
        body_pos = _parse_vec(body.get("pos"), default=np.zeros(3, dtype=np.float32))
        body_quat = _parse_vec(body.get("quat"), default=np.array([1, 0, 0, 0], dtype=np.float32))
        body_rot = _quat_wxyz_to_matrix(body_quat)
        offset_k = MUJOCO_TO_KIMODO @ body_pos
        rot_k = MUJOCO_TO_KIMODO @ body_rot @ MUJOCO_TO_KIMODO.T
        if joint_name in by_name:
            idx = by_name[joint_name]
            if idx != 0:
                local_offsets[idx] = offset_k
            rest_local_rotations[idx] = rot_k

        for child in body.findall("body"):
            walk(child)

    for body in worldbody.findall("body"):
        walk(body)
    return local_offsets, rest_local_rotations


def _parse_mesh_local_transforms(root: ET.Element) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mesh_file_by_name = {
        mesh.get("name"): mesh.get("file")
        for mesh in root.findall(".//asset/mesh")
        if mesh.get("name") and mesh.get("file")
    }
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for geom in root.findall(".//geom"):
        mesh_name = geom.get("mesh")
        mesh_file = mesh_file_by_name.get(mesh_name)
        if mesh_file is None or mesh_file in out:
            continue
        pos = _parse_vec(geom.get("pos"), default=np.zeros(3, dtype=np.float32))
        quat = _parse_vec(geom.get("quat"), default=np.array([1, 0, 0, 0], dtype=np.float32))
        rot = _quat_wxyz_to_matrix(quat)
        out[mesh_file] = (MUJOCO_TO_KIMODO @ pos, MUJOCO_TO_KIMODO @ rot @ MUJOCO_TO_KIMODO.T)
    return out


def _parse_qpos_joints(
    root: ET.Element,
    class_axes: dict[str, np.ndarray],
    class_limits: dict[str, tuple[float, float]],
) -> tuple[list[int], np.ndarray, np.ndarray, list[str]]:
    indices: list[int] = []
    axes: list[np.ndarray] = []
    limits: list[tuple[float, float]] = []
    names: list[str] = []
    by_name = {name: i for i, name in enumerate(JOINT_NAMES)}
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("g1.xml is missing a worldbody")

    for joint in worldbody.findall(".//joint"):
        name = joint.get("name")
        if not name or name == "floating_base_joint":
            continue
        skel_name = name.replace("_joint", "_skel")
        if skel_name not in by_name:
            continue
        axis = _joint_axis(joint, class_axes)
        axis_k = MUJOCO_TO_KIMODO @ axis
        norm = np.linalg.norm(axis_k)
        if norm <= 1e-8:
            continue
        axes.append(axis_k / norm)
        indices.append(by_name[skel_name])
        limits.append(_joint_limit(joint, class_limits))
        names.append(skel_name)
    return indices, np.asarray(axes), np.asarray(limits), names


def _load_link_meshes(mesh_dir: Path, mesh_transforms: dict[str, tuple[np.ndarray, np.ndarray]], *, dtype) -> tuple:
    vertices_by_link: list[np.ndarray] = []
    faces_by_link: list[np.ndarray] = []
    joint_indices: list[int] = []
    vertex_starts: list[int] = []
    vertex_counts: list[int] = []
    face_starts: list[int] = []
    face_counts: list[int] = []
    geom_positions: list[np.ndarray] = []
    geom_rotations: list[np.ndarray] = []
    names: list[str] = []
    vertex_offset = 0
    face_offset = 0
    by_name = {name: i for i, name in enumerate(JOINT_NAMES)}

    for joint_name, mesh_files in G1_MESH_JOINT_MAP.items():
        joint_idx = by_name[joint_name]
        for mesh_file in mesh_files:
            if mesh_file not in mesh_transforms:
                raise FileNotFoundError(f"G1 XML does not reference expected mesh: {mesh_file}")
            path = mesh_dir / mesh_file
            if not path.exists():
                raise FileNotFoundError(f"G1 mesh not found: {path}")
            vertices, faces = load_stl_mesh(path, dtype=dtype)
            vertices_by_link.append(vertices)
            faces_by_link.append(faces + vertex_offset)
            geom_pos, geom_rot = mesh_transforms[mesh_file]
            joint_indices.append(joint_idx)
            vertex_starts.append(vertex_offset)
            vertex_counts.append(vertices.shape[0])
            face_starts.append(face_offset)
            face_counts.append(faces.shape[0])
            geom_positions.append(geom_pos)
            geom_rotations.append(geom_rot)
            names.append(mesh_file)
            vertex_offset += vertices.shape[0]
            face_offset += faces.shape[0]

    if not vertices_by_link:
        raise FileNotFoundError(f"No G1 STL link meshes found in {mesh_dir}")
    link_data = {
        "joint_indices": joint_indices,
        "vertex_starts": vertex_starts,
        "vertex_counts": vertex_counts,
        "face_starts": face_starts,
        "face_counts": face_counts,
        "geom_positions": np.asarray(geom_positions),
        "geom_rotations": np.asarray(geom_rotations),
        "names": names,
    }
    return np.concatenate(vertices_by_link), np.concatenate(faces_by_link), link_data


def load_stl_mesh(path: Path, *, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if _looks_like_binary_stl(data):
        return _load_binary_stl(data, dtype=dtype)
    return _load_ascii_stl(data.decode("utf-8"), dtype=dtype)


def _load_ascii_stl(text: str, *, dtype) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(vertices) % 3 != 0 or not vertices:
        raise ValueError("ASCII STL contains no triangular facets")
    verts = np.asarray(vertices, dtype=dtype) @ MUJOCO_TO_KIMODO.T
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return verts, faces


def _load_binary_stl(data: bytes, *, dtype) -> tuple[np.ndarray, np.ndarray]:
    n_tri = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + n_tri * 50
    if len(data) < expected:
        raise ValueError("Binary STL is truncated")
    vertices = np.empty((n_tri * 3, 3), dtype=dtype)
    offset = 84
    for tri in range(n_tri):
        offset += 12
        for corner in range(3):
            vertices[tri * 3 + corner] = struct.unpack_from("<fff", data, offset)
            offset += 12
        offset += 2
    return vertices @ MUJOCO_TO_KIMODO.T, np.arange(n_tri * 3, dtype=np.int64).reshape(-1, 3)


def _looks_like_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    n_tri = struct.unpack_from("<I", data, 80)[0]
    return 84 + n_tri * 50 == len(data)


def _body_to_joint_name(body: ET.Element) -> str:
    joint = body.find("joint")
    joint_name = joint.get("name") if joint is not None else None
    if joint_name and joint_name != "floating_base_joint":
        return joint_name.replace("_joint", "_skel")
    name = body.get("name", "")
    if name == "pelvis":
        return "pelvis_skel"
    return name.removesuffix("_link") + "_skel"


def _joint_axis(joint: ET.Element, class_axes: dict[str, np.ndarray]) -> np.ndarray:
    axis = joint.get("axis")
    if axis:
        return np.asarray([float(x) for x in axis.split()], dtype=np.float32)
    class_name = joint.get("class")
    if class_name in class_axes:
        return class_axes[class_name]
    raise ValueError(f"Missing axis for G1 joint {joint.get('name')}")


def _joint_limit(joint: ET.Element, class_limits: dict[str, tuple[float, float]]) -> tuple[float, float]:
    limit = joint.get("range")
    if limit:
        lo, hi = [float(x) for x in limit.split()]
        return lo, hi
    class_name = joint.get("class")
    if class_name in class_limits:
        return class_limits[class_name]
    return -np.inf, np.inf


def _parse_vec(value: str | None, *, default: np.ndarray) -> np.ndarray:
    if value is None:
        return default.astype(np.float32, copy=True)
    return np.asarray([float(x) for x in value.split()], dtype=np.float32)


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
        dtype=np.float32,
    )
