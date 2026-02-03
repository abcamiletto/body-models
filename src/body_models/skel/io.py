from pathlib import Path

import numpy as np

from .. import config


def get_model_path(model_path: Path | str | None, gender: str) -> Path:
    if model_path is None:
        model_path = config.get_model_path("skel")

    if model_path is None:
        raise FileNotFoundError(
            "SKEL model not found. Download from https://skel.is.tue.mpg.de/ "
            "and run: body-models set skel /path/to/skel (only male/female supported)"
        )

    model_path = Path(model_path)

    if model_path.is_file():
        return model_path

    if model_path.is_dir():
        candidate = model_path / f"skel_{gender.lower()}.pkl"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"SKEL {gender} model not found in {model_path}")


def simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simplify mesh using quadric decimation.

    Args:
        vertices: [V, 3] vertex positions
        faces: [F, 3] face indices
        target_faces: target number of faces

    Returns:
        new_vertices: [V', 3] simplified vertex positions
        new_faces: [F', 3] simplified face indices
        vertex_map: [V'] index of nearest original vertex for each new vertex
    """
    import pyfqmr
    from scipy.spatial import KDTree

    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(vertices, faces)
    simplifier.simplify_mesh(target_count=target_faces, aggressiveness=7, preserve_border=True)
    new_vertices, new_faces, _ = simplifier.getMesh()

    new_vertices = np.asarray(new_vertices, dtype=np.float32)
    new_faces = np.asarray(new_faces, dtype=np.int32)

    # Find nearest original vertex for each new vertex (for attribute mapping)
    tree = KDTree(vertices)
    _, vertex_map = tree.query(new_vertices)
    vertex_map = np.asarray(vertex_map, dtype=np.int64)

    return new_vertices, new_faces, vertex_map
