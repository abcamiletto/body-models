# Derived from: https://github.com/naver/anny
# Original license: Apache 2.0 (https://github.com/naver/anny/blob/main/LICENSE)

import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from ..base import BodyModel
from ..utils import get_cache_dir
from . import core
from .io import EXCLUDED_PHENOTYPES, PHENOTYPE_LABELS, PHENOTYPE_VARIATIONS, get_model_path


class ANNY(BodyModel, nn.Module):
    """ANNY body model with phenotype-based morphology.

    Args:
        model_path: Path to ANNY model directory. Auto-downloads if None.
        cache_dir: Cache directory for preprocessed data.
        rig: Skeleton rig type ("default", "default_no_toes", "cmu_mb", "game_engine", "mixamo").
        topology: Mesh topology ("default" or "makehuman").
        all_phenotypes: Include race, cupsize, firmness phenotypes.
        extrapolate_phenotypes: Allow phenotype values outside [0, 1].
        simplify: Mesh simplification ratio. 1.0 = original mesh, 2.0 = half faces, etc.
            Note: Simplified meshes use triangular faces instead of quads.

    Forward API:
        forward_vertices(gender, age, muscle, weight, height, proportions, pose, ...)
        forward_skeleton(gender, age, muscle, weight, height, proportions, pose, ...)

        Phenotype parameters: [B] tensors in [0, 1]
        pose: [B, J, 3] axis-angle per joint
    """

    # Buffer type annotations
    template_vertices: Float[Tensor, "V 3"]
    blendshapes: Float[Tensor, "S V 3"]
    template_bone_heads: Float[Tensor, "J 3"]
    template_bone_tails: Float[Tensor, "J 3"]
    bone_heads_blendshapes: Float[Tensor, "S J 3"]
    bone_tails_blendshapes: Float[Tensor, "S J 3"]
    bone_rolls_rotmat: Float[Tensor, "J 3 3"]
    vertex_bone_weights: Float[Tensor, "V K"]
    vertex_bone_indices: Int[Tensor, "V K"]
    lbs_weights: Float[Tensor, "V J"]
    phenotype_mask: Float[Tensor, "S P"]
    _y_axis: Float[Tensor, "3"]
    _degenerate_rotation: Float[Tensor, "3 3"]
    _coord_rotation: Float[Tensor, "3 3"]
    _coord_translation: Float[Tensor, "3"]

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        cache_dir: Path | str | None = None,
        rig: str = "default",
        topology: str = "default",
        all_phenotypes: bool = False,
        extrapolate_phenotypes: bool = False,
        simplify: float = 1.0,
    ) -> None:
        assert rig in ("default", "default_no_toes", "cmu_mb", "game_engine", "mixamo")
        assert topology in ("default", "makehuman")
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"
        super().__init__()

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
            orig_dtype = data["template_vertices"].dtype

            # Convert quads to triangles: each quad (a,b,c,d) -> triangles (a,b,c), (a,c,d)
            quads = data["faces"]
            tri_faces = torch.cat([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], dim=0)

            # Simplify mesh
            target_faces = int(len(tri_faces) / simplify)
            vertices_np = data["template_vertices"].numpy()
            faces_np = tri_faces.numpy().astype(int)
            new_vertices, new_faces, vertex_map = _simplify_mesh(vertices_np, faces_np, target_faces)

            # Remap per-vertex attributes (preserve original dtype)
            data["template_vertices"] = torch.as_tensor(new_vertices, dtype=orig_dtype)
            data["blendshapes"] = data["blendshapes"][:, vertex_map]
            data["vertex_bone_weights"] = data["vertex_bone_weights"][vertex_map]
            data["vertex_bone_indices"] = data["vertex_bone_indices"][vertex_map]
            data["faces"] = torch.as_tensor(new_faces, dtype=torch.int64)

        # Register buffers
        dtype = data["template_vertices"].dtype
        for key in [
            "template_vertices",
            "blendshapes",
            "template_bone_heads",
            "template_bone_tails",
            "bone_heads_blendshapes",
            "bone_tails_blendshapes",
            "bone_rolls_rotmat",
            "vertex_bone_weights",
            "vertex_bone_indices",
            "phenotype_mask",
        ]:
            self.register_buffer(key, data[key], persistent=False)

        self.register_buffer("_y_axis", torch.tensor([0.0, 1.0, 0.0], dtype=dtype), persistent=False)
        self.register_buffer(
            "_degenerate_rotation", torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=dtype)), persistent=False
        )
        self.register_buffer(
            "_coord_rotation",
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=dtype),
            persistent=False,
        )
        self.register_buffer("_coord_translation", torch.tensor([0.0, 0.852, 0.0], dtype=dtype), persistent=False)

        # Precompute dense LBS weights for faster skinning
        V, J = data["vertex_bone_weights"].shape[0], len(data["bone_labels"])
        lbs_weights = torch.zeros(V, J, dtype=dtype)
        lbs_weights.scatter_(1, data["vertex_bone_indices"], data["vertex_bone_weights"])
        self.register_buffer("lbs_weights", lbs_weights, persistent=False)

        self._faces = data["faces"]
        self.bone_parents = data["bone_parents"]
        self.bone_labels = data["bone_labels"]
        self._kinematic_fronts = _build_kinematic_fronts(data["bone_parents"])
        self.extrapolate_phenotypes = extrapolate_phenotypes
        self.all_phenotypes = all_phenotypes
        self.phenotype_labels = (
            PHENOTYPE_LABELS if all_phenotypes else [x for x in PHENOTYPE_LABELS if x not in EXCLUDED_PHENOTYPES]
        )

        # Phenotype interpolation anchors
        self._anchors = nn.ParameterDict(
            {
                "age": nn.Parameter(torch.linspace(-1 / 3, 1.0, 5, dtype=dtype), requires_grad=False),
                **{
                    k: nn.Parameter(
                        torch.linspace(0.0, 1.0, len(PHENOTYPE_VARIATIONS[k]), dtype=dtype), requires_grad=False
                    )
                    for k in ["gender", "muscle", "weight", "height", "proportions", "cupsize", "firmness"]
                },
            }
        )

    @property
    def faces(self) -> Int[Tensor, "F _"]:
        """Face indices. Shape [F, 4] for quads (original) or [F, 3] for triangles (simplified)."""
        return self._faces

    @property
    def num_joints(self) -> int:
        return len(self.bone_labels)

    @property
    def num_vertices(self) -> int:
        return self.template_vertices.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        return self.template_vertices.dtype

    @property
    def skin_weights(self) -> Float[Tensor, "V J"]:
        return self.lbs_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.template_vertices @ self._coord_rotation.T + self._coord_translation

    def forward_vertices(
        self,
        gender: Float[Tensor, "B"],
        age: Float[Tensor, "B"],
        muscle: Float[Tensor, "B"],
        weight: Float[Tensor, "B"],
        height: Float[Tensor, "B"],
        proportions: Float[Tensor, "B"],
        pose: Float[Tensor, "B J 3"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B V 3"]:
        """Compute mesh vertices [B, V, 3]."""
        return core.forward_vertices(
            template_vertices=self.template_vertices,
            blendshapes=self.blendshapes,
            template_bone_heads=self.template_bone_heads,
            template_bone_tails=self.template_bone_tails,
            bone_heads_blendshapes=self.bone_heads_blendshapes,
            bone_tails_blendshapes=self.bone_tails_blendshapes,
            bone_rolls_rotmat=self.bone_rolls_rotmat,
            lbs_weights=self.lbs_weights,
            phenotype_mask=self.phenotype_mask,
            anchors=self._get_anchors_dict(),
            kinematic_fronts=self._kinematic_fronts,
            coord_rotation=self._coord_rotation,
            coord_translation=self._coord_translation,
            y_axis=self._y_axis,
            degenerate_rotation=self._degenerate_rotation,
            extrapolate_phenotypes=self.extrapolate_phenotypes,
            gender=gender,
            age=age,
            muscle=muscle,
            weight=weight,
            height=height,
            proportions=proportions,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
        )

    def forward_skeleton(
        self,
        gender: Float[Tensor, "B"],
        age: Float[Tensor, "B"],
        muscle: Float[Tensor, "B"],
        weight: Float[Tensor, "B"],
        height: Float[Tensor, "B"],
        proportions: Float[Tensor, "B"],
        pose: Float[Tensor, "B J 3"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B J 4 4"]:
        """Compute skeleton transforms [B, J, 4, 4]."""
        return core.forward_skeleton(
            template_bone_heads=self.template_bone_heads,
            template_bone_tails=self.template_bone_tails,
            bone_heads_blendshapes=self.bone_heads_blendshapes,
            bone_tails_blendshapes=self.bone_tails_blendshapes,
            bone_rolls_rotmat=self.bone_rolls_rotmat,
            phenotype_mask=self.phenotype_mask,
            anchors=self._get_anchors_dict(),
            kinematic_fronts=self._kinematic_fronts,
            coord_rotation=self._coord_rotation,
            coord_translation=self._coord_translation,
            y_axis=self._y_axis,
            degenerate_rotation=self._degenerate_rotation,
            extrapolate_phenotypes=self.extrapolate_phenotypes,
            gender=gender,
            age=age,
            muscle=muscle,
            weight=weight,
            height=height,
            proportions=proportions,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        """Get rest pose parameters."""
        device = self.template_vertices.device
        return {
            **{
                k: torch.full((batch_size,), 0.5, device=device, dtype=dtype)
                for k in ["gender", "age", "muscle", "weight", "height", "proportions"]
            },
            "pose": torch.zeros((batch_size, self.num_joints, 3), device=device, dtype=dtype),
            "global_rotation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
            "global_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }

    def _get_anchors_dict(self) -> dict[str, Tensor]:
        """Get anchors as a plain dict for core functions."""
        return {name: self._anchors[name].data for name in self._anchors}


def _build_kinematic_fronts(parents: list[int]) -> tuple[list[list[int]], list[list[int]]]:
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


# Data loading

_RIG_CONFIGS = {
    "default": ("rig.default.json", "weights.default.json"),
    "default_no_toes": ("rig.default_no_toes.json", "weights.default.json"),
    "cmu_mb": ("rig.cmu_mb.json", "weights.cmu_mb.json"),
    "game_engine": ("rig.game_engine.json", "weights.game_engine.json"),
    "mixamo": ("rig.mixamo.json", "weights.mixamo.json"),
}


def _load_data(data_dir: Path, cache_dir: Path, rig: str, eyes: bool, tongue: bool) -> dict:
    """Load ANNY model data (cached)."""
    cache_key = hashlib.md5(f"{rig}_{eyes}_{tongue}".encode()).hexdigest()
    cache_file = cache_dir / f"data_{cache_key}.pth"
    if cache_file.exists():
        return torch.load(cache_file, weights_only=True)

    dtype = torch.float32
    world_T = (
        0.1 * SO3.to_matrix(SO3.from_euler(torch.tensor([[torch.pi / 2, 0, 0]], dtype=dtype), convention="xyz"))[0]
    )

    # Load mesh
    mesh_path = data_dir / "data" / "mpfb2" / "3dobjs" / "base.obj"
    verts, uvs, groups = _load_obj(mesh_path, dtype)
    verts = verts @ world_T.T

    for g in groups.values():
        g["unique_verts"] = torch.unique(g["faces"].flatten())

    # Collect faces
    face_groups = ["body"] + (["helper-l-eye", "helper-r-eye"] if eyes else []) + (["helper-tongue"] if tongue else [])
    faces = torch.cat([groups[g]["faces"] for g in face_groups])
    face_uvs = torch.cat([groups[g]["face_uvs"] for g in face_groups])

    # Load rig
    rig_file, weights_file = _RIG_CONFIGS[rig]
    rig_dir = data_dir / "data" / "mpfb2" / "rigs" / "standard"
    rig_data = json.loads((rig_dir / rig_file).read_text())
    if rig == "mixamo":
        rig_data = rig_data["bones"]
    weights_data = json.loads((rig_dir / weights_file).read_text())

    bone_labels, bone_parents = _build_skeleton(rig_data)
    bone_indices, bone_weights = _build_skin_weights(weights_data, bone_labels, len(verts), dtype)

    # Load blendshapes
    blendshapes_dict = _load_blendshapes(data_dir, verts, world_T, dtype)
    blendshapes, phenotype_mask = _stack_blendshapes(blendshapes_dict, dtype)

    # Bone positions from blendshapes
    heads, tails, heads_bs, tails_bs, rolls = _compute_bone_data(
        verts, blendshapes, bone_labels, rig_data, groups, dtype
    )

    result = {
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
    torch.save(result, cache_file)
    return result


def _load_obj(path: Path, dtype: torch.dtype) -> tuple[Tensor, Tensor, dict]:
    """Load OBJ file."""
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
        g["faces"] = torch.tensor(g["faces"], dtype=torch.int64)
        g["face_uvs"] = (
            torch.tensor(g["face_uvs"], dtype=torch.int64) if g["face_uvs"] else torch.empty(0, dtype=torch.int64)
        )

    return (
        torch.tensor(verts, dtype=dtype),
        torch.tensor(uvs, dtype=dtype) if uvs else torch.empty(0, 2, dtype=dtype),
        groups,
    )


def _load_blendshapes(
    data_dir: Path, template: Float[Tensor, "V 3"], world_T: Float[Tensor, "3 3"], dtype: torch.dtype
) -> dict:
    """Load all phenotype blendshapes."""
    n = len(template)
    pv = PHENOTYPE_VARIATIONS
    macro = data_dir / "data" / "mpfb2" / "targets" / "macrodetails"
    breast = data_dir / "data" / "mpfb2" / "targets" / "breast"
    newborn_scale = torch.tensor([0.922, 0.922, 0.75], dtype=dtype)

    def load(path: Path, age: str) -> Tensor:
        bs = torch.zeros(n, 3, dtype=dtype)
        with gzip.open(path, "rt") as f:
            for line in f:
                p = line.split()
                bs[int(p[0])] = torch.tensor([float(x) for x in p[1:]], dtype=dtype)
        bs = bs @ world_T.T
        if age == "newborn":
            bs = newborn_scale * bs + ((newborn_scale - 1) / 3) * template
        return bs

    def age_file(a):
        return "baby" if a == "newborn" else a

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


def _stack_blendshapes(shapes: dict, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Stack blendshapes and build phenotype mask."""
    all_phens = [p for vs in PHENOTYPE_VARIATIONS.values() for p in vs]
    stacked, masks = [], []
    for keys, shape in shapes.items():
        stacked.append(shape)
        mask = torch.zeros(len(all_phens), dtype=dtype)
        for k in keys:
            mask[all_phens.index(k)] = 1
        masks.append(mask)
    return torch.stack(stacked), torch.stack(masks)


def _build_skeleton(rig_data: dict) -> tuple[list[str], list[int]]:
    """Build bone hierarchy from rig data."""
    root = next(k for k, v in rig_data.items() if v["parent"] == "")
    labels, parents = [], []

    def add(label, parent_id):
        labels.append(label)
        parents.append(parent_id)
        cur_id = len(labels) - 1
        for k, v in rig_data.items():
            if k not in labels and v["parent"] == label:
                add(k, cur_id)

    add(root, -1)
    return labels, parents


def _build_skin_weights(
    weights_data: dict, labels: list[str], n_verts: int, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Build sparse skin weight tensors."""
    indices = [[] for _ in range(n_verts)]
    weights = [[] for _ in range(n_verts)]

    for bone_id, label in enumerate(labels):
        for v_idx, w in weights_data["weights"][label]:
            indices[v_idx].append(bone_id)
            weights[v_idx].append(w)

    max_k = max(len(i) for i in indices)
    for i, w in zip(indices, weights):
        while len(i) < max_k:
            i.append(0)
            w.append(0.0)

    idx_t = torch.tensor(indices, dtype=torch.int64)
    w_t = torch.tensor(weights, dtype=dtype)
    w_t /= w_t.sum(dim=-1, keepdim=True)
    return idx_t, w_t


def _compute_bone_data(
    verts: Float[Tensor, "V 3"],
    shapes: Float[Tensor, "S V 3"],
    labels: list,
    rig: dict,
    groups: dict,
    dtype: torch.dtype,
):
    """Compute bone head/tail positions and blendshapes."""

    def get_verts(data):
        s = data["strategy"]
        if s == "VERTEX":
            return [data["vertex_index"]]
        if s == "CUBE":
            return groups[data["cube_name"]]["unique_verts"].tolist()
        if s == "MEAN":
            return data["vertex_indices"]
        raise ValueError(s)

    heads, tails, heads_bs, tails_bs, rolls = [], [], [], [], []
    for label in labels:
        h_idx = torch.tensor(get_verts(rig[label]["head"]), dtype=torch.int64)
        t_idx = torch.tensor(get_verts(rig[label]["tail"]), dtype=torch.int64)
        heads.append(verts[h_idx].mean(0))
        tails.append(verts[t_idx].mean(0))
        heads_bs.append(shapes[:, h_idx].mean(1))
        tails_bs.append(shapes[:, t_idx].mean(1))
        rolls.append(rig[label]["roll"])

    euler = torch.zeros(len(rolls), 3, dtype=dtype)
    euler[:, 1] = torch.tensor(rolls, dtype=dtype)
    rolls_mat = SO3.to_matrix(SO3.from_euler(euler, convention="xyz"))

    return torch.stack(heads), torch.stack(tails), torch.stack(heads_bs, dim=1), torch.stack(tails_bs, dim=1), rolls_mat


def _edit_mesh_faces(
    faces: Int[Tensor, "F 4"], uvs: Int[Tensor, "F 4"]
) -> tuple[Int[Tensor, "F2 4"], Int[Tensor, "F2 4"]]:
    """Edit MakeHuman mesh topology (ear caps)."""
    discard = torch.cat([torch.arange(1778, 1794), torch.arange(8450, 8466)]).to(faces.dtype)
    keep = ~torch.isin(faces, discard).any(dim=1)

    v2uv = {}
    for i in torch.nonzero(~keep, as_tuple=False).squeeze(1):
        for v, uv in zip(faces[i], uvs[i]):
            v2uv[v.item()] = uv.item()

    caps = torch.tensor(
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
    cap_uvs = torch.tensor([v2uv[v.item()] for v in caps.flatten()]).reshape_as(caps)

    return torch.cat([faces[keep], caps]), torch.cat([uvs[keep], cap_uvs])


def _simplify_mesh(
    vertices: Float[np.ndarray, "V 3"],
    faces: Int[np.ndarray, "F 3"],
    target_faces: int,
) -> tuple[Float[np.ndarray, "V2 3"], Int[np.ndarray, "F2 3"], Int[np.ndarray, "V2"]]:
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
