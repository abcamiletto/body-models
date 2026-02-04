from pathlib import Path

import numpy as np

from .. import config
from ..utils import download_and_extract, get_cache_dir

ANNY_URL = "https://github.com/naver/anny/archive/refs/heads/main.zip"

PHENOTYPE_VARIATIONS = {
    "race": ["african", "asian", "caucasian"],
    "gender": ["male", "female"],
    "age": ["newborn", "baby", "child", "young", "old"],
    "muscle": ["minmuscle", "averagemuscle", "maxmuscle"],
    "weight": ["minweight", "averageweight", "maxweight"],
    "height": ["minheight", "maxheight"],
    "proportions": ["idealproportions", "uncommonproportions"],
    "cupsize": ["mincup", "averagecup", "maxcup"],
    "firmness": ["minfirmness", "averagefirmness", "maxfirmness"],
}
PHENOTYPE_LABELS = [k for k in PHENOTYPE_VARIATIONS if k != "race"] + PHENOTYPE_VARIATIONS["race"]
EXCLUDED_PHENOTYPES = ["cupsize", "firmness"] + PHENOTYPE_VARIATIONS["race"]


def get_model_path(model_path: Path | str | None = None) -> Path:
    if model_path is None:
        model_path = config.get_model_path("anny")

    if model_path is not None:
        model_path = Path(model_path)
        if model_path.exists():
            return model_path
        raise FileNotFoundError(f"ANNY model path {model_path} does not exist")

    cache_path = get_cache_dir() / "anny"
    if (cache_path / "data" / "mpfb2").exists():
        return cache_path

    return download_model()


def download_model() -> Path:
    cache_dir = get_cache_dir() / "anny"
    print(f"Downloading ANNY model to {cache_dir}...")
    download_and_extract(url=ANNY_URL, dest=cache_dir, extract_subdir="anny-main/")
    print("Done")
    return cache_dir


def build_kinematic_fronts(parents: list[int]) -> tuple[list[list[int]], list[list[int]]]:
    """Group joints by depth for parallel forward kinematics."""
    n = len(parents)
    assigned = [False] * n
    level = [i for i in range(n) if parents[i] < 0]
    indices, parent_ids = [], []

    while level:
        indices.append(level)
        parent_ids.append([parents[i] for i in level])
        for j in level:
            assigned[j] = True
        level = [i for i in range(n) if not assigned[i] and parents[i] in level]

    return indices, parent_ids


def build_anchors(dtype=np.float32) -> dict[str, np.ndarray]:
    """Build phenotype interpolation anchors."""
    return {
        "age": np.linspace(-1 / 3, 1.0, 5, dtype=dtype),
        **{
            k: np.linspace(0.0, 1.0, len(PHENOTYPE_VARIATIONS[k]), dtype=dtype)
            for k in ["gender", "muscle", "weight", "height", "proportions", "cupsize", "firmness"]
        },
    }


def load_model_data_numpy(
    model_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    rig: str = "default",
    topology: str = "default",
    simplify: float = 1.0,
    dtype=np.float32,
) -> dict:
    """Load ANNY model data as numpy arrays.

    Uses torch internally for data loading (cached), then converts to numpy.
    """
    import torch

    # Import torch-based loading function
    from .torch import _load_data, _edit_mesh_faces, _simplify_mesh, _build_kinematic_fronts

    resolved_path = get_model_path(model_path)
    cache_dir = Path(cache_dir) if cache_dir else get_cache_dir() / "anny" / "preprocessed"

    data = _load_data(resolved_path, cache_dir, rig=rig, eyes=True, tongue=True)

    # Apply topology edits
    if topology == "default":
        data["faces"], data["face_uvs"] = _edit_mesh_faces(data["faces"], data["face_uvs"])

    # Remove unattached vertices
    used_verts = torch.unique(data["faces"].flatten(), sorted=True)
    remap = torch.full((len(data["template_vertices"]),), -1, dtype=torch.int64)
    remap[used_verts] = torch.arange(len(used_verts))

    data["template_vertices"] = data["template_vertices"][used_verts]
    data["vertex_bone_weights"] = data["vertex_bone_weights"][used_verts]
    data["vertex_bone_indices"] = data["vertex_bone_indices"][used_verts]
    data["blendshapes"] = data["blendshapes"][:, used_verts]
    data["faces"] = remap[data["faces"].flatten()].reshape(data["faces"].shape)

    # Trim zero-weight bone columns
    while (data["vertex_bone_weights"].min(dim=-1).values == 0).all():
        keep = data["vertex_bone_weights"].argmin(dim=-1, keepdim=True)
        mask = torch.arange(data["vertex_bone_weights"].shape[1])[None, :] != keep
        data["vertex_bone_weights"] = data["vertex_bone_weights"][mask].reshape(len(used_verts), -1)
        data["vertex_bone_indices"] = data["vertex_bone_indices"][mask].reshape(len(used_verts), -1)

    # Apply mesh simplification if requested
    if simplify > 1.0:
        quads = data["faces"]
        tri_faces = torch.cat([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], dim=0)

        target_faces = int(len(tri_faces) / simplify)
        vertices_np = data["template_vertices"].numpy()
        faces_np = tri_faces.numpy().astype(int)
        new_vertices, new_faces, vertex_map = _simplify_mesh(vertices_np, faces_np, target_faces)

        data["template_vertices"] = torch.as_tensor(new_vertices, dtype=data["template_vertices"].dtype)
        data["blendshapes"] = data["blendshapes"][:, vertex_map]
        data["vertex_bone_weights"] = data["vertex_bone_weights"][vertex_map]
        data["vertex_bone_indices"] = data["vertex_bone_indices"][vertex_map]
        data["faces"] = torch.as_tensor(new_faces, dtype=torch.int64)

    # Precompute dense LBS weights
    V, J = data["vertex_bone_weights"].shape[0], len(data["bone_labels"])
    lbs_weights = torch.zeros(V, J, dtype=data["template_vertices"].dtype)
    lbs_weights.scatter_(1, data["vertex_bone_indices"], data["vertex_bone_weights"])

    # Build kinematic fronts
    kinematic_fronts = _build_kinematic_fronts(data["bone_parents"])

    # Convert to numpy
    result = {
        "template_vertices": data["template_vertices"].numpy().astype(dtype),
        "blendshapes": data["blendshapes"].numpy().astype(dtype),
        "template_bone_heads": data["template_bone_heads"].numpy().astype(dtype),
        "template_bone_tails": data["template_bone_tails"].numpy().astype(dtype),
        "bone_heads_blendshapes": data["bone_heads_blendshapes"].numpy().astype(dtype),
        "bone_tails_blendshapes": data["bone_tails_blendshapes"].numpy().astype(dtype),
        "bone_rolls_rotmat": data["bone_rolls_rotmat"].numpy().astype(dtype),
        "phenotype_mask": data["phenotype_mask"].numpy().astype(dtype),
        "lbs_weights": lbs_weights.numpy().astype(dtype),
        "faces": data["faces"].numpy(),
        "bone_labels": data["bone_labels"],
        "bone_parents": data["bone_parents"],
        "kinematic_fronts": kinematic_fronts,
    }

    return result
