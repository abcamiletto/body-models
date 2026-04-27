"""I/O utilities for the MyoFullBody musculoskeletal model.

The upstream model is the ``musclemimic_models`` package from
``amathislab/musclemimic_models``. We download a pinned snapshot of its
``model/`` directory (MJCF + STL meshes) and parse it without depending on the
``mujoco`` runtime.
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from nanomanifold import SO3

from .. import config
from ..utils import download_and_extract, get_cache_dir

# MUJOCO_TO_KIMODO maps MuJoCo's Z-up world to body-models Y-up. MyoFullBody's
# OpenSim-derived bodies still come out with their lateral axis on Z, so an
# additional Ry(+90°) puts left/right on ±X to match SMPL/G1 and the rendering
# pipeline (X = lateral, Y = up, Z = depth).
_RY_90 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
_MUJOCO_TO_KIMODO_BARE = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32
)
MUJOCO_TO_KIMODO = (_RY_90 @ _MUJOCO_TO_KIMODO_BARE).astype(np.float32)
MUSCLEMIMIC_REPO_ZIP = "https://github.com/amathislab/musclemimic_models/archive/refs/heads/main.zip"
MAIN_XML_RELPATH = Path("body") / "myofullbody.xml"
ROOT_BODY_NAME = "Full Body"


# ----------------------------------------------------------------------------
# Path resolution / download
# ----------------------------------------------------------------------------


def get_model_path(model_path: Path | str | None = None) -> Path:
    """Resolve a directory containing the upstream MuscleMimic ``model/`` tree."""
    if model_path is None:
        model_path = config.get_model_path("myofullbody")

    if model_path is not None:
        path = Path(model_path)
        return _validate_model_path(path)

    cache_path = get_cache_dir() / "myofullbody"
    if (cache_path / MAIN_XML_RELPATH).exists():
        return cache_path

    return download_model()


def download_model() -> Path:
    """Download the upstream ``musclemimic_models/model/`` snapshot to cache."""
    cache_dir = get_cache_dir() / "myofullbody"
    print(f"Downloading MyoFullBody model to {cache_dir}...")
    download_and_extract(
        url=MUSCLEMIMIC_REPO_ZIP,
        dest=cache_dir,
        extract_subdir="musclemimic_models-main/musclemimic_models/model/",
    )
    print("Done")
    return cache_dir


def _validate_model_path(path: Path) -> Path:
    xml_path = path / MAIN_XML_RELPATH
    if not xml_path.exists():
        raise FileNotFoundError(f"MyoFullBody main XML not found: {xml_path}")
    return path


# ----------------------------------------------------------------------------
# Top-level loader
# ----------------------------------------------------------------------------


def load_model_data(model_path: Path | str | None = None, *, dtype=np.float32) -> dict:
    """Parse ``body/myofullbody.xml`` (with ``<include>`` resolution) plus link STLs."""
    model_dir = get_model_path(model_path)
    xml_path = model_dir / MAIN_XML_RELPATH

    root = _parse_with_includes(xml_path)
    mesh_files = _parse_mesh_assets(root)
    class_defaults = _parse_class_defaults(root)

    body_xml = _find_root_body_in_root(root, xml_path)

    body_records: list[dict] = []
    qpos_records: list[dict] = []
    link_records: list[dict] = []
    site_records: list[dict] = []
    _walk_body(body_xml, parent_idx=-1, parent_class=None,
               bodies=body_records, qpos=qpos_records, links=link_records,
               sites=site_records, defaults=class_defaults, is_root=True)

    joint_names = [b["name"] for b in body_records]
    parents = [b["parent"] for b in body_records]
    local_offsets = np.stack([b["pos"] for b in body_records])
    rest_local_rotations = np.stack([b["rot"] for b in body_records])

    body_qpos_starts: list[int] = []
    body_qpos_counts: list[int] = []
    cursor = 0
    for body in body_records:
        body_qpos_starts.append(cursor)
        body_qpos_counts.append(body["qpos_count"])
        cursor += body["qpos_count"]

    qpos_joint_names = [q["name"] for q in qpos_records]
    qpos_joint_axes = _stack_or_empty(qpos_records, "axis", (0, 3))
    qpos_joint_anchors = _stack_or_empty(qpos_records, "anchor", (0, 3))
    qpos_joint_types = [q["type"] for q in qpos_records]
    qpos_joint_limits = _stack_or_empty(qpos_records, "range", (0, 2))
    hinge_mask = np.asarray([t == "hinge" for t in qpos_joint_types], dtype=np.float32)
    slide_mask = np.asarray([t == "slide" for t in qpos_joint_types], dtype=np.float32)

    vertices, faces, link_meta = _build_link_meshes(
        link_records, mesh_files, model_dir, dtype=dtype,
    )

    site_names = [s["name"] for s in site_records]
    site_positions = _stack_or_empty(site_records, "pos", (0, 3))
    site_body_indices = [s["body"] for s in site_records]
    tendons = _parse_tendons(root, site_names, class_defaults)

    return {
        "joint_names": joint_names,
        "parents": parents,
        "local_offsets": local_offsets.astype(dtype),
        "rest_local_rotations": rest_local_rotations.astype(dtype),
        "qpos_joint_names": qpos_joint_names,
        "qpos_joint_axes": qpos_joint_axes.astype(dtype),
        "qpos_joint_anchors": qpos_joint_anchors.astype(dtype),
        "qpos_joint_types": qpos_joint_types,
        "qpos_joint_limits": qpos_joint_limits.astype(dtype),
        "hinge_mask": hinge_mask.astype(dtype),
        "slide_mask": slide_mask.astype(dtype),
        "body_qpos_starts": body_qpos_starts,
        "body_qpos_counts": body_qpos_counts,
        "vertices": vertices.astype(dtype),
        "faces": faces.astype(np.int64),
        "link_joint_indices": link_meta["joint_indices"],
        "link_vertex_starts": link_meta["vertex_starts"],
        "link_vertex_counts": link_meta["vertex_counts"],
        "link_face_starts": link_meta["face_starts"],
        "link_face_counts": link_meta["face_counts"],
        "link_geom_positions": link_meta["geom_positions"].astype(dtype),
        "link_geom_rotations": link_meta["geom_rotations"].astype(dtype),
        "link_names": link_meta["names"],
        "site_names": site_names,
        "site_positions": site_positions.astype(dtype),
        "site_body_indices": site_body_indices,
        "tendons": tendons,
    }


def _stack_or_empty(records: list[dict], key: str, empty_shape: tuple[int, ...]) -> np.ndarray:
    if not records:
        return np.zeros(empty_shape, dtype=np.float32)
    return np.stack([r[key] for r in records])


# ----------------------------------------------------------------------------
# Recursive XML <include> resolution
# ----------------------------------------------------------------------------


def _parse_with_includes(path: Path) -> ET.Element:
    """Parse an MJCF file, recursively inlining ``<include file="..."/>`` elements."""
    tree = ET.parse(path)
    root = tree.getroot()
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
            sub_tree = ET.parse(include_path)
            sub_root = sub_tree.getroot()
            _inline_includes(sub_root, include_path.parent, visited | {include_path})
            new_children.extend(list(sub_root))
        else:
            _inline_includes(child, base_dir, visited)
            new_children.append(child)
    element[:] = new_children


# ----------------------------------------------------------------------------
# Class defaults & mesh assets
# ----------------------------------------------------------------------------


_DEFAULT_JOINT = {
    "axis": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "range": (-np.inf, np.inf),
    "type": "hinge",
    "tendon_width": 0.005,
}


def _parse_class_defaults(root: ET.Element) -> dict[str, dict]:
    """Return per-class defaults (joint axis/range/type + tendon width) from ``<default class=...>``."""
    out: dict[str, dict] = {}

    def visit(element: ET.Element, parent: dict) -> None:
        local = dict(parent)
        joint = element.find("joint")
        if joint is not None:
            axis = joint.get("axis")
            if axis:
                local["axis"] = _parse_vec(axis, default=local.get("axis"))
            limit = joint.get("range")
            if limit:
                local["range"] = tuple(float(x) for x in limit.split())
            joint_type = joint.get("type")
            if joint_type:
                local["type"] = joint_type
        tendon = element.find("tendon")
        if tendon is not None and tendon.get("width"):
            local["tendon_width"] = float(tendon.get("width"))
        class_name = element.get("class")
        if class_name:
            out[class_name] = local
        for child in element.findall("default"):
            visit(child, local)

    base = dict(_DEFAULT_JOINT)
    for top in root.findall("default"):
        visit(top, base)
    return out


def _parse_tendons(
    root: ET.Element, site_names: list[str], class_defaults: dict[str, dict]
) -> list[dict]:
    """Collect ``<spatial>`` tendons as polyline lists of site indices.

    Wrap geoms (``<geom geom=...>`` inside the tendon) are skipped — we render
    straight via-point segments only — and any tendon referencing an unknown
    site is dropped.
    """
    site_index = {name: i for i, name in enumerate(site_names)}
    out: list[dict] = []
    for spatial in root.findall(".//tendon/spatial"):
        default = class_defaults.get(spatial.get("class") or "", _DEFAULT_JOINT)
        width = float(spatial.get("width") or default.get("tendon_width", 0.005))

        refs = [s.get("site") for s in spatial.findall("site")]
        if any(r is None or r not in site_index for r in refs) or len(refs) < 2:
            continue
        out.append({
            "name": spatial.get("name") or f"tendon_{len(out)}",
            "site_indices": [site_index[r] for r in refs],
            "width": width,
        })
    return out


def _parse_mesh_assets(root: ET.Element) -> dict[str, tuple[str, np.ndarray]]:
    """Return ``{mesh_name: (file_path, scale_xyz)}`` from all ``<asset><mesh>`` entries.

    MyoFullBody mirrors right-side STLs to make left-side bones (e.g.
    ``<mesh name="humerus_l" file="meshes/humerus.stl" scale="1 1 -1"/>``), so we
    must propagate the per-mesh scale into the loader.
    """
    out: dict[str, tuple[str, np.ndarray]] = {}
    for mesh in root.findall(".//asset/mesh"):
        name = mesh.get("name")
        file = mesh.get("file")
        if not name or not file:
            continue
        scale = _parse_vec(mesh.get("scale"), default=np.ones(3, dtype=np.float32))
        out[name] = (file, scale)
    return out


# ----------------------------------------------------------------------------
# Body / joint walker
# ----------------------------------------------------------------------------


def _find_root_body_in_root(root: ET.Element, xml_path: Path) -> ET.Element:
    """Pick the body that owns the model's freejoint root.

    After ``<include>`` resolution the merged document can contain multiple
    ``<worldbody>`` siblings (one per source file); we scan all of them and
    pick the first body that hosts a ``<freejoint>`` or matches the expected
    root name, falling back to the first body otherwise.
    """
    fallback: ET.Element | None = None
    for worldbody in root.findall("worldbody"):
        for body in worldbody.findall("body"):
            if body.find("freejoint") is not None or body.get("name") == ROOT_BODY_NAME:
                return body
            if fallback is None:
                fallback = body
    if fallback is None:
        raise ValueError(f"{xml_path} has no <body> in any <worldbody>")
    return fallback


def _walk_body(
    elem: ET.Element,
    parent_idx: int,
    parent_class: str | None,
    bodies: list[dict],
    qpos: list[dict],
    links: list[dict],
    sites: list[dict],
    defaults: dict[str, dict],
    is_root: bool,
) -> None:
    name = elem.get("name") or f"body_{len(bodies)}"
    childclass = elem.get("childclass") or parent_class

    # Freejoint root: the freejoint qpos overrides the body's XML pos/quat at
    # runtime, so we collapse the root frame to identity and let
    # global_translation/global_rotation drive it from the public API.
    if is_root:
        pos = np.zeros(3, dtype=np.float32)
        rot = np.eye(3, dtype=np.float32)
    else:
        raw_pos = _parse_vec(elem.get("pos"), default=np.zeros(3, dtype=np.float32))
        raw_rot = _parse_orientation(elem)
        pos = MUJOCO_TO_KIMODO @ raw_pos
        rot = MUJOCO_TO_KIMODO @ raw_rot @ MUJOCO_TO_KIMODO.T

    body_idx = len(bodies)
    body_record = {"name": name, "parent": parent_idx, "pos": pos, "rot": rot, "qpos_count": 0}
    bodies.append(body_record)

    for joint in elem.findall("joint"):
        cls_default = defaults.get(joint.get("class") or childclass or "", _DEFAULT_JOINT)
        joint_type = joint.get("type") or cls_default.get("type", "hinge")
        # Ball/freejoint-typed entries fall outside our hinge+slide chain composition.
        if joint_type not in {"hinge", "slide"}:
            continue
        axis_raw = _parse_vec(
            joint.get("axis"),
            default=np.asarray(cls_default.get("axis", _DEFAULT_JOINT["axis"]), dtype=np.float32),
        )
        axis_raw = axis_raw / max(float(np.linalg.norm(axis_raw)), 1e-12)
        anchor_raw = _parse_vec(joint.get("pos"), default=np.zeros(3, dtype=np.float32))
        rng = joint.get("range")
        if rng:
            lo, hi = (float(x) for x in rng.split())
        else:
            lo, hi = cls_default.get("range", _DEFAULT_JOINT["range"])
        qpos.append({
            "name": joint.get("name") or f"joint_{len(qpos)}",
            "axis": (MUJOCO_TO_KIMODO @ axis_raw).astype(np.float32),
            "anchor": (MUJOCO_TO_KIMODO @ anchor_raw).astype(np.float32),
            "type": joint_type,
            "range": np.asarray([lo, hi], dtype=np.float32),
        })
        body_record["qpos_count"] += 1

    for geom in elem.findall("geom"):
        mesh = geom.get("mesh")
        if not mesh:
            continue
        gpos_raw = _parse_vec(geom.get("pos"), default=np.zeros(3, dtype=np.float32))
        grot_raw = _parse_orientation(geom)
        links.append({
            "body": body_idx,
            "mesh_name": mesh,
            "geom_name": geom.get("name") or mesh,
            "geom_pos": (MUJOCO_TO_KIMODO @ gpos_raw).astype(np.float32),
            "geom_rot": (MUJOCO_TO_KIMODO @ grot_raw @ MUJOCO_TO_KIMODO.T).astype(np.float32),
        })

    for site in elem.findall("site"):
        name = site.get("name")
        if not name:
            continue
        spos_raw = _parse_vec(site.get("pos"), default=np.zeros(3, dtype=np.float32))
        sites.append({
            "name": name,
            "body": body_idx,
            "pos": (MUJOCO_TO_KIMODO @ spos_raw).astype(np.float32),
        })

    for child in elem.findall("body"):
        _walk_body(child, body_idx, childclass, bodies, qpos, links, sites, defaults, is_root=False)


# ----------------------------------------------------------------------------
# Mesh loading
# ----------------------------------------------------------------------------


def _build_link_meshes(
    link_records: list[dict],
    mesh_files: dict[str, str],
    model_dir: Path,
    *,
    dtype,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not link_records:
        raise FileNotFoundError("No <geom mesh=\"...\"/> entries found in MyoFullBody XML")

    vertices_chunks: list[np.ndarray] = []
    faces_chunks: list[np.ndarray] = []
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

    for link in link_records:
        asset = mesh_files.get(link["mesh_name"])
        if asset is None:
            raise FileNotFoundError(f"MyoFullBody mesh asset missing: {link['mesh_name']}")
        mesh_file, scale = asset
        path = (model_dir / mesh_file).resolve()
        if not path.exists():
            raise FileNotFoundError(f"MyoFullBody mesh file not found: {path}")
        verts, faces = load_stl_mesh(path, dtype=dtype, scale=scale)
        local_faces = faces + vertex_offset

        vertices_chunks.append(verts)
        faces_chunks.append(local_faces)
        joint_indices.append(link["body"])
        vertex_starts.append(vertex_offset)
        vertex_counts.append(verts.shape[0])
        face_starts.append(face_offset)
        face_counts.append(local_faces.shape[0])
        geom_positions.append(link["geom_pos"])
        geom_rotations.append(link["geom_rot"])
        names.append(link["geom_name"])
        vertex_offset += verts.shape[0]
        face_offset += local_faces.shape[0]

    return (
        np.concatenate(vertices_chunks),
        np.concatenate(faces_chunks),
        {
            "joint_indices": joint_indices,
            "vertex_starts": vertex_starts,
            "vertex_counts": vertex_counts,
            "face_starts": face_starts,
            "face_counts": face_counts,
            "geom_positions": np.asarray(geom_positions, dtype=np.float32),
            "geom_rotations": np.asarray(geom_rotations, dtype=np.float32),
            "names": names,
        },
    )


def load_stl_mesh(
    path: Path, *, dtype=np.float32, scale: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load an STL into kimodo coordinates, applying an optional per-mesh ``scale``.

    ``scale`` is the MJCF ``<mesh scale="...">`` triple, applied in the STL's own
    (mujoco) frame before rotating into kimodo. Reflective scales (``det < 0``)
    flip triangle winding so outward normals stay consistent.
    """
    data = path.read_bytes()
    if _looks_like_binary_stl(data):
        raw_verts, faces = _load_binary_stl_raw(data, dtype=dtype)
    else:
        raw_verts, faces = _load_ascii_stl_raw(data.decode("utf-8"), dtype=dtype)
    if scale is not None and not np.allclose(scale, 1.0):
        raw_verts = raw_verts * scale.astype(raw_verts.dtype, copy=False)
        if float(np.prod(scale)) < 0.0:
            faces = faces[:, ::-1].copy()
    return raw_verts @ MUJOCO_TO_KIMODO.T, faces


def _load_ascii_stl_raw(text: str, *, dtype) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(vertices) % 3 != 0 or not vertices:
        raise ValueError("ASCII STL contains no triangular facets")
    return (
        np.asarray(vertices, dtype=dtype),
        np.arange(len(vertices), dtype=np.int64).reshape(-1, 3),
    )


_BINARY_STL_TRI_DTYPE = np.dtype(
    [("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]
)


def _load_binary_stl_raw(data: bytes, *, dtype) -> tuple[np.ndarray, np.ndarray]:
    n_tri = struct.unpack_from("<I", data, 80)[0]
    if len(data) < 84 + n_tri * 50:
        raise ValueError("Binary STL is truncated")
    triangles = np.frombuffer(data, dtype=_BINARY_STL_TRI_DTYPE, count=n_tri, offset=84)
    vertices = triangles["vertices"].reshape(-1, 3).astype(dtype, copy=False)
    return vertices, np.arange(n_tri * 3, dtype=np.int64).reshape(-1, 3)


def _looks_like_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    n_tri = struct.unpack_from("<I", data, 80)[0]
    return 84 + n_tri * 50 == len(data)


# ----------------------------------------------------------------------------
# Small parsing helpers
# ----------------------------------------------------------------------------


def _parse_vec(value: str | None, *, default: np.ndarray) -> np.ndarray:
    if value is None:
        return default.astype(np.float32, copy=True)
    return np.asarray([float(x) for x in value.split()], dtype=np.float32)


def _parse_orientation(elem: ET.Element) -> np.ndarray:
    """Return a 3x3 rotation matrix from a ``quat`` or ``euler`` attribute.

    MuJoCo's default ``eulerseq="xyz"`` means *intrinsic* XYZ rotations
    (``R = Rx(a) @ Ry(b) @ Rz(c)``); in nanomanifold this is the uppercase
    ``"XYZ"`` convention, distinct from lowercase ``"xyz"`` (extrinsic).
    """
    quat = elem.get("quat")
    if quat:
        q = np.asarray([float(x) for x in quat.split()], dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        return SO3.conversions.from_quat_to_rotmat(q, convention="wxyz", xp=np).astype(np.float32)
    euler = elem.get("euler")
    if euler:
        e = np.asarray([float(x) for x in euler.split()], dtype=np.float32)
        return SO3.conversions.from_euler_to_rotmat(e, convention="XYZ", xp=np).astype(np.float32)
    return np.eye(3, dtype=np.float32)
