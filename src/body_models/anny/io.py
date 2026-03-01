import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from nanomanifold import SO3

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

_RIG_CONFIGS = {
    "default": ("rig.default.json", "weights.default.json"),
    "default_no_toes": ("rig.default_no_toes.json", "weights.default.json"),
    "cmu_mb": ("rig.cmu_mb.json", "weights.cmu_mb.json"),
    "game_engine": ("rig.game_engine.json", "weights.game_engine.json"),
    "mixamo": ("rig.mixamo.json", "weights.mixamo.json"),
}


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
    download_and_extract(url=ANNY_URL, dest=cache_dir, extract_subdir="anny-main/src/anny/")
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
    """Load ANNY model data as numpy arrays."""
    resolved_path = get_model_path(model_path)
    cache_dir = Path(cache_dir) if cache_dir else get_cache_dir() / "anny" / "preprocessed"
    data = _load_data_numpy(resolved_path, cache_dir, rig=rig, eyes=True, tongue=True, dtype=dtype)

    # Apply topology edits
    if topology == "default":
        data["faces"], data["face_uvs"] = _edit_mesh_faces(data["faces"], data["face_uvs"])

    # Remove unattached vertices
    used_verts = np.unique(data["faces"].reshape(-1))
    remap = np.full((len(data["template_vertices"]),), -1, dtype=np.int64)
    remap[used_verts] = np.arange(len(used_verts), dtype=np.int64)

    data["template_vertices"] = data["template_vertices"][used_verts]
    data["vertex_bone_weights"] = data["vertex_bone_weights"][used_verts]
    data["vertex_bone_indices"] = data["vertex_bone_indices"][used_verts]
    data["blendshapes"] = data["blendshapes"][:, used_verts]
    data["faces"] = remap[data["faces"].flatten()].reshape(data["faces"].shape)

    # Trim zero-weight bone columns
    while np.all(np.min(data["vertex_bone_weights"], axis=-1) == 0):
        keep = np.argmin(data["vertex_bone_weights"], axis=-1)
        mask = np.ones_like(data["vertex_bone_weights"], dtype=bool)
        mask[np.arange(mask.shape[0]), keep] = False
        data["vertex_bone_weights"] = data["vertex_bone_weights"][mask].reshape(len(used_verts), -1)
        data["vertex_bone_indices"] = data["vertex_bone_indices"][mask].reshape(len(used_verts), -1)

    # Apply mesh simplification if requested
    if simplify > 1.0:
        quads = data["faces"]
        tri_faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)
        target_faces = int(len(tri_faces) / simplify)
        new_vertices, new_faces, vertex_map = _simplify_mesh(data["template_vertices"], tri_faces.astype(int), target_faces)

        data["template_vertices"] = new_vertices.astype(data["template_vertices"].dtype)
        data["blendshapes"] = data["blendshapes"][:, vertex_map]
        data["vertex_bone_weights"] = data["vertex_bone_weights"][vertex_map]
        data["vertex_bone_indices"] = data["vertex_bone_indices"][vertex_map]
        data["faces"] = new_faces.astype(np.int64)

    # Precompute dense LBS weights
    V, J = data["vertex_bone_weights"].shape[0], len(data["bone_labels"])
    lbs_weights = np.zeros((V, J), dtype=data["template_vertices"].dtype)
    rows = np.arange(V)[:, None]
    lbs_weights[rows, data["vertex_bone_indices"]] = data["vertex_bone_weights"]

    # Build kinematic fronts
    kinematic_fronts = build_kinematic_fronts(data["bone_parents"])

    return {
        "template_vertices": data["template_vertices"].astype(dtype),
        "blendshapes": data["blendshapes"].astype(dtype),
        "template_bone_heads": data["template_bone_heads"].astype(dtype),
        "template_bone_tails": data["template_bone_tails"].astype(dtype),
        "bone_heads_blendshapes": data["bone_heads_blendshapes"].astype(dtype),
        "bone_tails_blendshapes": data["bone_tails_blendshapes"].astype(dtype),
        "bone_rolls_rotmat": data["bone_rolls_rotmat"].astype(dtype),
        "phenotype_mask": data["phenotype_mask"].astype(dtype),
        "lbs_weights": lbs_weights.astype(dtype),
        "faces": data["faces"],
        "bone_labels": data["bone_labels"],
        "bone_parents": data["bone_parents"],
        "kinematic_fronts": kinematic_fronts,
    }


def _cache_file_stem(rig: str, eyes: bool, tongue: bool) -> str:
    cache_key = hashlib.md5(f"{rig}_{eyes}_{tongue}".encode()).hexdigest()
    return f"data_{cache_key}"


def _load_data_numpy(
    data_dir: Path,
    cache_dir: Path,
    rig: str,
    eyes: bool,
    tongue: bool,
    dtype: np.dtype = np.float32,
) -> dict:
    """Load ANNY model data with NumPy, optionally reading legacy torch caches via ptloader."""
    stem = _cache_file_stem(rig, eyes, tongue)
    cache_npz = cache_dir / f"{stem}.npz"
    cache_pth = cache_dir / f"{stem}.pth"

    if cache_npz.exists():
        return _load_npz_cache(cache_npz)

    if cache_pth.exists():
        try:
            from ptloader import load as ptload
            data = ptload(cache_pth, weights_only=True)
            # Store converted cache in a torch-free format for future loads.
            cache_dir.mkdir(parents=True, exist_ok=True)
            _save_npz_cache(cache_npz, data)
            return data
        except ImportError:
            pass

    world_T = 0.1 * SO3.conversions.from_euler_to_matrix(
        np.array([[np.pi / 2, 0, 0]], dtype=dtype),
        convention="xyz",
    )[0]

    mesh_path = data_dir / "data" / "mpfb2" / "3dobjs" / "base.obj"
    verts, _uvs, groups = _load_obj(mesh_path, dtype)
    verts = verts @ world_T.T

    for g in groups.values():
        g["unique_verts"] = np.unique(g["faces"].reshape(-1))

    face_groups = ["body"] + (["helper-l-eye", "helper-r-eye"] if eyes else []) + (["helper-tongue"] if tongue else [])
    faces = np.concatenate([groups[g]["faces"] for g in face_groups], axis=0)
    face_uvs = np.concatenate([groups[g]["face_uvs"] for g in face_groups], axis=0)

    rig_file, weights_file = _RIG_CONFIGS[rig]
    rig_dir = data_dir / "data" / "mpfb2" / "rigs" / "standard"
    rig_data = json.loads((rig_dir / rig_file).read_text())
    if rig == "mixamo":
        rig_data = rig_data["bones"]
    weights_data = json.loads((rig_dir / weights_file).read_text())

    bone_labels, bone_parents = _build_skeleton(rig_data)
    bone_indices, bone_weights = _build_skin_weights(weights_data, bone_labels, len(verts), dtype)

    blendshapes_dict = _load_blendshapes(data_dir, verts, world_T, dtype)
    blendshapes, phenotype_mask = _stack_blendshapes(blendshapes_dict, dtype)

    heads, tails, heads_bs, tails_bs, rolls = _compute_bone_data(
        verts,
        blendshapes,
        bone_labels,
        rig_data,
        groups,
        dtype,
    )

    data = {
        "template_vertices": verts,
        "faces": faces,
        "face_uvs": face_uvs,
        "blendshapes": blendshapes,
        "phenotype_mask": phenotype_mask,
        "template_bone_heads": heads,
        "template_bone_tails": tails,
        "bone_heads_blendshapes": heads_bs,
        "bone_tails_blendshapes": tails_bs,
        "bone_rolls_rotmat": rolls,
        "bone_labels": bone_labels,
        "bone_parents": bone_parents,
        "vertex_bone_weights": bone_weights,
        "vertex_bone_indices": bone_indices,
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    _save_npz_cache(cache_npz, data)
    return data


def _save_npz_cache(cache_file: Path, data: dict) -> None:
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    arrays["bone_labels"] = np.asarray(data["bone_labels"], dtype=object)
    arrays["bone_parents"] = np.asarray(data["bone_parents"], dtype=np.int64)
    np.savez_compressed(cache_file, **arrays)


def _load_npz_cache(cache_file: Path) -> dict:
    raw = np.load(cache_file, allow_pickle=True)
    out = dict(raw)
    out["bone_labels"] = [str(x) for x in out["bone_labels"].tolist()]
    out["bone_parents"] = [int(x) for x in out["bone_parents"].tolist()]
    return out


def _load_obj(path: Path, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray, dict]:
    verts, uvs, groups, cur = [], [], {}, None

    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif cmd == "vt":
            uvs.append([float(x) for x in parts[1:3]])
        elif cmd == "g":
            cur = parts[1]
            groups[cur] = {"faces": [], "face_uvs": []}
        elif cmd == "f" and cur:
            fv, ft = [], []
            for p in parts[1:]:
                idx = p.split("/")
                fv.append(int(idx[0]) - 1)
                if len(idx) > 1 and idx[1]:
                    ft.append(int(idx[1]) - 1)
            groups[cur]["faces"].append(fv)
            if ft:
                groups[cur]["face_uvs"].append(ft)

    for g in groups.values():
        g["faces"] = np.asarray(g["faces"], dtype=np.int64)
        g["face_uvs"] = np.asarray(g["face_uvs"], dtype=np.int64) if g["face_uvs"] else np.empty(0, dtype=np.int64)

    uv_arr = np.asarray(uvs, dtype=dtype) if uvs else np.empty((0, 2), dtype=dtype)
    return np.asarray(verts, dtype=dtype), uv_arr, groups


def _load_blendshapes(data_dir: Path, template: np.ndarray, world_T: np.ndarray, dtype: np.dtype) -> dict:
    n = len(template)
    pv = PHENOTYPE_VARIATIONS
    macro = data_dir / "data" / "mpfb2" / "targets" / "macrodetails"
    breast = data_dir / "data" / "mpfb2" / "targets" / "breast"
    newborn_scale = np.asarray([0.922, 0.922, 0.75], dtype=dtype)

    def load(path: Path, age: str) -> np.ndarray:
        bs = np.zeros((n, 3), dtype=dtype)
        with gzip.open(path, "rt") as f:
            for line in f:
                p = line.split()
                bs[int(p[0])] = np.asarray([float(x) for x in p[1:]], dtype=dtype)
        bs = bs @ world_T.T
        if age == "newborn":
            bs = newborn_scale * bs + ((newborn_scale - 1) / 3) * template
        return bs

    def age_file(age: str) -> str:
        return "baby" if age == "newborn" else age

    shapes = {}
    for g, a, m, w in itertools.product(pv["gender"], pv["age"], pv["muscle"], pv["weight"]):
        shapes[(g, a, m, w)] = load(macro / f"universal-{g}-{age_file(a)}-{m}-{w}.target.gz", a)
    for r, g, a in itertools.product(pv["race"], pv["gender"], pv["age"]):
        shapes[(r, g, a)] = load(macro / f"{r}-{g}-{age_file(a)}.target.gz", a)
    for g, a, m, w, h in itertools.product(pv["gender"], pv["age"], pv["muscle"], pv["weight"], pv["height"]):
        shapes[(g, a, m, w, h)] = load(macro / "height" / f"{g}-{age_file(a)}-{m}-{w}-{h}.target.gz", a)
    for g, a, m, w, p in itertools.product(pv["gender"], pv["age"], pv["muscle"], pv["weight"], pv["proportions"]):
        if a not in ("newborn", "baby"):
            shapes[(g, a, m, w, p)] = load(macro / "proportions" / f"{g}-{a}-{m}-{w}-{p}.target.gz", a)
    for a, m, w, c, f in itertools.product(pv["age"], pv["muscle"], pv["weight"], pv["cupsize"], pv["firmness"]):
        path = breast / f"female-{a}-{m}-{w}-{c}-{f}.target.gz"
        if path.exists():
            shapes[("female", a, m, w, c, f)] = load(path, a)

    return shapes


def _stack_blendshapes(shapes: dict, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    all_phens = [p for vals in PHENOTYPE_VARIATIONS.values() for p in vals]
    ph2idx = {p: i for i, p in enumerate(all_phens)}
    stacked, masks = [], []
    for keys, shape in shapes.items():
        stacked.append(shape)
        mask = np.zeros(len(all_phens), dtype=dtype)
        for k in keys:
            mask[ph2idx[k]] = 1
        masks.append(mask)
    return np.stack(stacked), np.stack(masks)


def _build_skeleton(rig_data: dict) -> tuple[list[str], list[int]]:
    root = next(k for k, v in rig_data.items() if v["parent"] == "")
    labels, parents = [], []

    def add(label: str, parent_id: int) -> None:
        labels.append(label)
        parents.append(parent_id)
        cur_id = len(labels) - 1
        for k, v in rig_data.items():
            if k not in labels and v["parent"] == label:
                add(k, cur_id)

    add(root, -1)
    return labels, parents


def _build_skin_weights(
    weights_data: dict,
    labels: list[str],
    n_verts: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    indices = [[] for _ in range(n_verts)]
    weights = [[] for _ in range(n_verts)]

    for bone_id, label in enumerate(labels):
        for v_idx, w in weights_data["weights"][label]:
            indices[v_idx].append(bone_id)
            weights[v_idx].append(w)

    max_k = max(len(i) for i in indices)
    for i, w in zip(indices, weights, strict=True):
        while len(i) < max_k:
            i.append(0)
            w.append(0.0)

    idx_arr = np.asarray(indices, dtype=np.int64)
    w_arr = np.asarray(weights, dtype=dtype)
    w_arr /= w_arr.sum(axis=-1, keepdims=True)
    return idx_arr, w_arr


def _compute_bone_data(
    verts: np.ndarray,
    shapes: np.ndarray,
    labels: list[str],
    rig: dict,
    groups: dict,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def get_verts(data: dict) -> list[int]:
        strategy = data["strategy"]
        if strategy == "VERTEX":
            return [data["vertex_index"]]
        if strategy == "CUBE":
            return groups[data["cube_name"]]["unique_verts"].tolist()
        if strategy == "MEAN":
            return data["vertex_indices"]
        raise ValueError(strategy)

    heads, tails, heads_bs, tails_bs, rolls = [], [], [], [], []
    for label in labels:
        h_idx = np.asarray(get_verts(rig[label]["head"]), dtype=np.int64)
        t_idx = np.asarray(get_verts(rig[label]["tail"]), dtype=np.int64)
        heads.append(verts[h_idx].mean(axis=0))
        tails.append(verts[t_idx].mean(axis=0))
        heads_bs.append(shapes[:, h_idx].mean(axis=1))
        tails_bs.append(shapes[:, t_idx].mean(axis=1))
        rolls.append(rig[label]["roll"])

    euler = np.zeros((len(rolls), 3), dtype=dtype)
    euler[:, 1] = np.asarray(rolls, dtype=dtype)
    rolls_mat = SO3.conversions.from_euler_to_matrix(euler, convention="xyz", xp=np)

    return (
        np.stack(heads),
        np.stack(tails),
        np.stack(heads_bs, axis=1),
        np.stack(tails_bs, axis=1),
        rolls_mat,
    )


def _edit_mesh_faces(faces: np.ndarray, uvs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    discard = np.concatenate([np.arange(1778, 1794), np.arange(8450, 8466)]).astype(faces.dtype)
    keep = ~np.isin(faces, discard).any(axis=1)

    v2uv = {}
    for i in np.nonzero(~keep)[0]:
        for v, uv in zip(faces[i], uvs[i], strict=True):
            v2uv[int(v)] = int(uv)

    caps = np.asarray(
        [
            [8437, 8438, 8439, 8440],
            [8436, 8437, 8440, 8441],
            [8435, 8436, 8441, 8442],
            [8434, 8435, 8442, 8443],
            [8449, 8434, 8443, 8444],
            [8448, 8449, 8444, 8445],
            [8447, 8448, 8445, 8446],
            [1762, 1771, 1770, 1763],
            [1763, 1770, 1769, 1764],
            [1764, 1769, 1768, 1765],
            [1765, 1768, 1767, 1766],
            [1762, 1777, 1772, 1771],
            [1777, 1776, 1773, 1772],
            [1776, 1775, 1774, 1773],
        ],
        dtype=faces.dtype,
    )
    cap_uvs = np.asarray([v2uv[int(v)] for v in caps.reshape(-1)], dtype=uvs.dtype).reshape(caps.shape)

    return np.concatenate([faces[keep], caps], axis=0), np.concatenate([uvs[keep], cap_uvs], axis=0)


def _simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pyfqmr
    from scipy.spatial import KDTree

    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(vertices, faces)
    simplifier.simplify_mesh(target_count=target_faces, aggressiveness=7, preserve_border=True)
    new_vertices, new_faces, _ = simplifier.getMesh()

    new_vertices = np.asarray(new_vertices, dtype=np.float32)
    new_faces = np.asarray(new_faces, dtype=np.int32)

    tree = KDTree(vertices)
    _, vertex_map = tree.query(new_vertices)
    vertex_map = np.asarray(vertex_map, dtype=np.int64)

    return new_vertices, new_faces, vertex_map
