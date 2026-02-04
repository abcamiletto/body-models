"""NumPy backend for ANNY model."""

from pathlib import Path

import numpy as np

from . import core
from .io import (
    EXCLUDED_PHENOTYPES,
    PHENOTYPE_LABELS,
    build_anchors,
    load_model_data_numpy,
)

# Re-export conversion functions from core
from_native_args = core.from_native_args
to_native_outputs = core.to_native_outputs

__all__ = ["ANNY", "from_native_args", "to_native_outputs"]


class ANNY:
    """ANNY body model with NumPy backend.

    Args:
        model_path: Path to ANNY model directory. Auto-downloads if None.
        cache_dir: Cache directory for preprocessed data.
        rig: Skeleton rig type ("default", "default_no_toes", "cmu_mb", "game_engine", "mixamo").
        topology: Mesh topology ("default" or "makehuman").
        all_phenotypes: Include race, cupsize, firmness phenotypes.
        extrapolate_phenotypes: Allow phenotype values outside [0, 1].
        simplify: Mesh simplification ratio. 1.0 = original mesh, 2.0 = half faces, etc.
    """

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

        data = load_model_data_numpy(
            model_path=model_path,
            cache_dir=cache_dir,
            rig=rig,
            topology=topology,
            simplify=simplify,
            dtype=np.float32,
        )

        # Store model data
        self.template_vertices = data["template_vertices"]
        self.blendshapes = data["blendshapes"]
        self.template_bone_heads = data["template_bone_heads"]
        self.template_bone_tails = data["template_bone_tails"]
        self.bone_heads_blendshapes = data["bone_heads_blendshapes"]
        self.bone_tails_blendshapes = data["bone_tails_blendshapes"]
        self.bone_rolls_rotmat = data["bone_rolls_rotmat"]
        self.phenotype_mask = data["phenotype_mask"]
        self.lbs_weights = data["lbs_weights"]
        self._faces = data["faces"]
        self.bone_labels = data["bone_labels"]
        self.bone_parents = data["bone_parents"]
        self._kinematic_fronts = data["kinematic_fronts"]

        # Constants
        dtype = np.float32
        self._y_axis = np.array([0.0, 1.0, 0.0], dtype=dtype)
        self._degenerate_rotation = np.diag(np.array([1.0, -1.0, -1.0], dtype=dtype))
        self._coord_rotation = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=dtype
        )
        self._coord_translation = np.array([0.0, 0.852, 0.0], dtype=dtype)

        # Phenotype anchors
        self._anchors = build_anchors(dtype=dtype)

        self.extrapolate_phenotypes = extrapolate_phenotypes
        self.all_phenotypes = all_phenotypes
        self.phenotype_labels = (
            PHENOTYPE_LABELS if all_phenotypes else [x for x in PHENOTYPE_LABELS if x not in EXCLUDED_PHENOTYPES]
        )

    @property
    def faces(self) -> np.ndarray:
        """Face indices. Shape [F, 4] for quads (original) or [F, 3] for triangles (simplified)."""
        return self._faces

    @property
    def num_joints(self) -> int:
        return len(self.bone_labels)

    @property
    def num_vertices(self) -> int:
        return self.template_vertices.shape[0]

    @property
    def skin_weights(self) -> np.ndarray:
        return self.lbs_weights

    def forward_vertices(
        self,
        gender: np.ndarray,
        age: np.ndarray,
        muscle: np.ndarray,
        weight: np.ndarray,
        height: np.ndarray,
        proportions: np.ndarray,
        pose: np.ndarray,
        global_rotation: np.ndarray | None = None,
        global_translation: np.ndarray | None = None,
    ) -> np.ndarray:
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
            anchors=self._anchors,
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
        gender: np.ndarray,
        age: np.ndarray,
        muscle: np.ndarray,
        weight: np.ndarray,
        height: np.ndarray,
        proportions: np.ndarray,
        pose: np.ndarray,
        global_rotation: np.ndarray | None = None,
        global_translation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute skeleton transforms [B, J, 4, 4]."""
        return core.forward_skeleton(
            template_bone_heads=self.template_bone_heads,
            template_bone_tails=self.template_bone_tails,
            bone_heads_blendshapes=self.bone_heads_blendshapes,
            bone_tails_blendshapes=self.bone_tails_blendshapes,
            bone_rolls_rotmat=self.bone_rolls_rotmat,
            phenotype_mask=self.phenotype_mask,
            anchors=self._anchors,
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

    def get_rest_pose(self, batch_size: int = 1, dtype=np.float32) -> dict[str, np.ndarray]:
        """Get rest pose parameters."""
        return {
            **{
                k: np.full((batch_size,), 0.5, dtype=dtype)
                for k in ["gender", "age", "muscle", "weight", "height", "proportions"]
            },
            "pose": np.zeros((batch_size, self.num_joints, 3), dtype=dtype),
            "global_rotation": np.zeros((batch_size, 3), dtype=dtype),
            "global_translation": np.zeros((batch_size, 3), dtype=dtype),
        }
