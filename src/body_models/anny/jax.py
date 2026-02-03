"""JAX backend for ANNY model using Flax NNX."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

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


class ANNY(nnx.Module):
    """ANNY body model with JAX/Flax NNX backend.

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

        # Store model data as nnx.Variable for proper pytree handling
        self.template_vertices = nnx.Variable(jnp.asarray(data["template_vertices"]))
        self.blendshapes = nnx.Variable(jnp.asarray(data["blendshapes"]))
        self.template_bone_heads = nnx.Variable(jnp.asarray(data["template_bone_heads"]))
        self.template_bone_tails = nnx.Variable(jnp.asarray(data["template_bone_tails"]))
        self.bone_heads_blendshapes = nnx.Variable(jnp.asarray(data["bone_heads_blendshapes"]))
        self.bone_tails_blendshapes = nnx.Variable(jnp.asarray(data["bone_tails_blendshapes"]))
        self.bone_rolls_rotmat = nnx.Variable(jnp.asarray(data["bone_rolls_rotmat"]))
        self.phenotype_mask = nnx.Variable(jnp.asarray(data["phenotype_mask"]))
        self.lbs_weights = nnx.Variable(jnp.asarray(data["lbs_weights"]))
        self._faces = nnx.Variable(jnp.asarray(data["faces"]))
        self.bone_labels = data["bone_labels"]
        self.bone_parents = data["bone_parents"]
        self._kinematic_fronts = data["kinematic_fronts"]

        # Constants
        dtype = jnp.float32
        self._y_axis = nnx.Variable(jnp.array([0.0, 1.0, 0.0], dtype=dtype))
        self._degenerate_rotation = nnx.Variable(jnp.diag(jnp.array([1.0, -1.0, -1.0], dtype=dtype)))
        self._coord_rotation = nnx.Variable(
            jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=dtype)
        )
        self._coord_translation = nnx.Variable(jnp.array([0.0, 0.852, 0.0], dtype=dtype))

        # Phenotype anchors (stored as individual nnx.Variable)
        anchors_np = build_anchors(dtype=np.float32)
        self._anchor_age = nnx.Variable(jnp.asarray(anchors_np["age"]))
        self._anchor_gender = nnx.Variable(jnp.asarray(anchors_np["gender"]))
        self._anchor_muscle = nnx.Variable(jnp.asarray(anchors_np["muscle"]))
        self._anchor_weight = nnx.Variable(jnp.asarray(anchors_np["weight"]))
        self._anchor_height = nnx.Variable(jnp.asarray(anchors_np["height"]))
        self._anchor_proportions = nnx.Variable(jnp.asarray(anchors_np["proportions"]))
        self._anchor_cupsize = nnx.Variable(jnp.asarray(anchors_np["cupsize"]))
        self._anchor_firmness = nnx.Variable(jnp.asarray(anchors_np["firmness"]))

        self.extrapolate_phenotypes = extrapolate_phenotypes
        self.all_phenotypes = all_phenotypes
        self.phenotype_labels = (
            PHENOTYPE_LABELS if all_phenotypes else [x for x in PHENOTYPE_LABELS if x not in EXCLUDED_PHENOTYPES]
        )

    @property
    def faces(self) -> jnp.ndarray:
        """Face indices. Shape [F, 4] for quads (original) or [F, 3] for triangles (simplified)."""
        return self._faces[...]

    @property
    def num_joints(self) -> int:
        return len(self.bone_labels)

    @property
    def num_vertices(self) -> int:
        return self.template_vertices[...].shape[0]

    @property
    def skin_weights(self) -> jnp.ndarray:
        return self.lbs_weights[...]

    def _get_anchors_dict(self) -> dict[str, jnp.ndarray]:
        """Get anchors as plain arrays for core functions."""
        return {
            "age": self._anchor_age[...],
            "gender": self._anchor_gender[...],
            "muscle": self._anchor_muscle[...],
            "weight": self._anchor_weight[...],
            "height": self._anchor_height[...],
            "proportions": self._anchor_proportions[...],
            "cupsize": self._anchor_cupsize[...],
            "firmness": self._anchor_firmness[...],
        }

    def forward_vertices(
        self,
        gender: jnp.ndarray,
        age: jnp.ndarray,
        muscle: jnp.ndarray,
        weight: jnp.ndarray,
        height: jnp.ndarray,
        proportions: jnp.ndarray,
        pose: jnp.ndarray,
        global_rotation: jnp.ndarray | None = None,
        global_translation: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute mesh vertices [B, V, 3]."""
        return core.forward_vertices(
            template_vertices=self.template_vertices[...],
            blendshapes=self.blendshapes[...],
            template_bone_heads=self.template_bone_heads[...],
            template_bone_tails=self.template_bone_tails[...],
            bone_heads_blendshapes=self.bone_heads_blendshapes[...],
            bone_tails_blendshapes=self.bone_tails_blendshapes[...],
            bone_rolls_rotmat=self.bone_rolls_rotmat[...],
            lbs_weights=self.lbs_weights[...],
            phenotype_mask=self.phenotype_mask[...],
            anchors=self._get_anchors_dict(),
            kinematic_fronts=self._kinematic_fronts,
            coord_rotation=self._coord_rotation[...],
            coord_translation=self._coord_translation[...],
            y_axis=self._y_axis[...],
            degenerate_rotation=self._degenerate_rotation[...],
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
        gender: jnp.ndarray,
        age: jnp.ndarray,
        muscle: jnp.ndarray,
        weight: jnp.ndarray,
        height: jnp.ndarray,
        proportions: jnp.ndarray,
        pose: jnp.ndarray,
        global_rotation: jnp.ndarray | None = None,
        global_translation: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute skeleton transforms [B, J, 4, 4]."""
        return core.forward_skeleton(
            template_bone_heads=self.template_bone_heads[...],
            template_bone_tails=self.template_bone_tails[...],
            bone_heads_blendshapes=self.bone_heads_blendshapes[...],
            bone_tails_blendshapes=self.bone_tails_blendshapes[...],
            bone_rolls_rotmat=self.bone_rolls_rotmat[...],
            phenotype_mask=self.phenotype_mask[...],
            anchors=self._get_anchors_dict(),
            kinematic_fronts=self._kinematic_fronts,
            coord_rotation=self._coord_rotation[...],
            coord_translation=self._coord_translation[...],
            y_axis=self._y_axis[...],
            degenerate_rotation=self._degenerate_rotation[...],
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

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jnp.ndarray]:
        """Get rest pose parameters."""
        return {
            **{
                k: jnp.full((batch_size,), 0.5, dtype=dtype)
                for k in ["gender", "age", "muscle", "weight", "height", "proportions"]
            },
            "pose": jnp.zeros((batch_size, self.num_joints, 3), dtype=dtype),
            "global_rotation": jnp.zeros((batch_size, 3), dtype=dtype),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }
