"""STL mesh loading helpers."""

from pathlib import Path
import struct

import numpy as np


def load_stl_mesh(
    path: Path,
    *,
    coord: np.ndarray | None = None,
    dtype=np.float32,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an STL mesh, optionally applying MJCF scale and a coordinate transform."""
    data = path.read_bytes()
    if _looks_like_binary_stl(data):
        vertices, faces = _load_binary_stl_raw(data, dtype=dtype)
    else:
        vertices, faces = _load_ascii_stl_raw(data.decode("utf-8"), dtype=dtype)

    if scale is not None and not np.allclose(scale, 1.0):
        vertices = vertices * scale.astype(vertices.dtype, copy=False)
        if float(np.prod(scale)) < 0.0:
            faces = faces[:, ::-1].copy()

    if coord is not None:
        vertices = vertices @ coord.T
    return vertices, faces


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


_BINARY_STL_TRI_DTYPE = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])


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
