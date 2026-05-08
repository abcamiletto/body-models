"""JAX frontend for ANNY."""

from pathlib import Path

import jax.numpy as jnp
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import common
from body_models.anny.backends import jax as backend
from body_models.anny.io import EXCLUDED_PHENOTYPES, PHENOTYPE_LABELS, load_model_data_numpy
from body_models.base import BodyModel
from body_models.rotations import VALID_ROTATION_TYPES, RotationType

__all__ = ["ANNY"]


class ANNY(BodyModel):
    """ANNY body model with JAX backend."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rig: str = "default",
        topology: str = "default",
        all_phenotypes: bool = False,
        extrapolate_phenotypes: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        if rig not in ("default", "default_no_toes", "cmu_mb", "game_engine", "mixamo"):
            raise ValueError(f"Invalid rig: {rig}")
        if topology not in ("default", "makehuman"):
            raise ValueError(f"Invalid topology: {topology}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")

        data = load_model_data_numpy(model_path, rig=rig, topology=topology, simplify=simplify)
        self.weights = common.jaxify(data)
        self.extrapolate_phenotypes = extrapolate_phenotypes
        self.all_phenotypes = all_phenotypes
        self.rotation_type = rotation_type
        self.phenotype_labels = (
            PHENOTYPE_LABELS if all_phenotypes else [x for x in PHENOTYPE_LABELS if x not in EXCLUDED_PHENOTYPES]
        )

    @property
    def faces(self) -> Int[jnp.ndarray, "F _"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.bone_labels)

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.bone_labels)

    @property
    def num_vertices(self) -> int:
        return self.weights.template_vertices.shape[0]

    @property
    def skin_weights(self) -> Float[jnp.ndarray, "V J"]:
        return self.weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[jnp.ndarray, "V 3"]:
        return self.weights.template_vertices

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        gender: Float[jnp.ndarray, "B"],
        age: Float[jnp.ndarray, "B"],
        muscle: Float[jnp.ndarray, "B"],
        weight: Float[jnp.ndarray, "B"],
        height: Float[jnp.ndarray, "B"],
        proportions: Float[jnp.ndarray, "B"],
        pose: Float[jnp.ndarray, "B J N"] | Float[jnp.ndarray, "B J 3 3"],
        global_rotation: Float[jnp.ndarray, "B N"] | Float[jnp.ndarray, "B 3 3"] | None = None,
        global_translation: Float[jnp.ndarray, "B 3"] | None = None,
        vertex_indices: list[int] | None = None,
    ) -> Float[jnp.ndarray, "B V 3"]:
        return backend.forward_vertices(
            weights=self.weights,
            gender=gender,
            age=age,
            muscle=muscle,
            weight=weight,
            height=height,
            proportions=proportions,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
            extrapolate_phenotypes=self.extrapolate_phenotypes,
        )

    def forward_skeleton(
        self,
        gender: Float[jnp.ndarray, "B"],
        age: Float[jnp.ndarray, "B"],
        muscle: Float[jnp.ndarray, "B"],
        weight: Float[jnp.ndarray, "B"],
        height: Float[jnp.ndarray, "B"],
        proportions: Float[jnp.ndarray, "B"],
        pose: Float[jnp.ndarray, "B J N"] | Float[jnp.ndarray, "B J 3 3"],
        global_rotation: Float[jnp.ndarray, "B N"] | Float[jnp.ndarray, "B 3 3"] | None = None,
        global_translation: Float[jnp.ndarray, "B 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[jnp.ndarray, "B J 4 4"]:
        return backend.forward_skeleton(
            weights=self.weights,
            gender=gender,
            age=age,
            muscle=muscle,
            weight=weight,
            height=height,
            proportions=proportions,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
            extrapolate_phenotypes=self.extrapolate_phenotypes,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jnp.ndarray]:
        pose_ref = jnp.zeros((batch_size,), dtype=dtype)
        return {
            **{
                name: jnp.full((batch_size,), 0.5, dtype=dtype)
                for name in ["gender", "age", "muscle", "weight", "height", "proportions"]
            },
            "pose": SO3.identity_as(
                pose_ref,
                batch_dims=(batch_size, self.num_joints),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_rotation": SO3.identity_as(
                pose_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }
