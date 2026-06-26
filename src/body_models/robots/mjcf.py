"""Small MJCF parsing helpers used by robot model loaders."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from jaxtyping import Float
from nanomanifold import SO3


def parse_xml(path: Path, *, inline_includes: bool = False) -> ET.Element:
    root = ET.parse(path).getroot()
    if inline_includes:
        _inline_includes(root, path.parent, set())
    return root


def _inline_includes(element: ET.Element, base_dir: Path, visited: set[Path]) -> None:
    children = list(element)
    new_children: list[ET.Element] = []
    for child in children:
        if child.tag == "include":
            file_attr = child.get("file")
            if not file_attr:
                continue
            include_path = (base_dir / file_attr).resolve()
            if include_path in visited:
                raise RuntimeError(f"Cyclic <include> at {include_path}")
            sub_root = ET.parse(include_path).getroot()
            _inline_includes(sub_root, include_path.parent, visited | {include_path})
            new_children.extend(list(sub_root))
            continue

        _inline_includes(child, base_dir, visited)
        new_children.append(child)
    element[:] = new_children


def parse_vec(value: str | None, *, default: Float[np.ndarray, "..."], size: int | None = None) -> np.ndarray:
    if value is None:
        return default.astype(default.dtype, copy=True)
    parsed = np.asarray([float(x) for x in value.split()], dtype=default.dtype)
    if size is not None and parsed.shape != (size,):
        raise ValueError(f"Expected vector with {size} values, got {value!r}")
    return parsed


def quat_wxyz_to_matrix(q: Float[np.ndarray, "4"]) -> Float[np.ndarray, "3 3"]:
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


def parse_orientation(element: ET.Element) -> Float[np.ndarray, "3 3"]:
    quat = element.get("quat")
    if quat:
        return quat_wxyz_to_matrix(parse_vec(quat, default=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)))

    euler = element.get("euler")
    if euler:
        angles = parse_vec(euler, default=np.zeros(3, dtype=np.float32), size=3)
        return SO3.conversions.from_euler_to_rotmat(angles, convention="XYZ", xp=np).astype(np.float32)

    return np.eye(3, dtype=np.float32)


def mesh_base_dir(root: ET.Element, xml_path: Path) -> Path:
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    return (xml_path.parent / meshdir).resolve() if meshdir else xml_path.parent.resolve()


def mesh_files_by_name(root: ET.Element) -> dict[str, str]:
    return {
        mesh.get("name"): mesh.get("file")
        for mesh in root.findall(".//asset/mesh")
        if mesh.get("name") and mesh.get("file")
    }


def mesh_assets(root: ET.Element) -> dict[str, tuple[str, np.ndarray]]:
    out: dict[str, tuple[str, np.ndarray]] = {}
    for mesh in root.findall(".//asset/mesh"):
        name = mesh.get("name")
        file = mesh.get("file")
        if not name or not file:
            continue
        out[name] = (file, parse_vec(mesh.get("scale"), default=np.ones(3, dtype=np.float32), size=3))
    return out


def joint_defaults(root: ET.Element) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, float]]]:
    axes: dict[str, np.ndarray] = {}
    limits: dict[str, tuple[float, float]] = {}
    for default in root.findall(".//default"):
        class_name = default.get("class")
        joint = default.find("joint")
        if not class_name or joint is None:
            continue
        if joint.get("axis"):
            axes[class_name] = parse_vec(joint.get("axis"), default=np.zeros(3, dtype=np.float32), size=3)
        if joint.get("range"):
            lo, hi = (float(x) for x in joint.get("range", "").split())
            limits[class_name] = (lo, hi)
    return axes, limits
