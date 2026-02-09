"""JAX backend for SKEL model using Flax NNX."""

import pickle as pkl
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Float, Int
from scipy import sparse

from ..base import BodyModel
from . import core
from .io import get_model_path, simplify_mesh

__all__ = ["SKEL", "from_native_args", "to_native_outputs"]


class SKEL(BodyModel, nnx.Module):
    """SKEL body model with JAX/Flax NNX backend.

    Args:
        gender: One of "male" or "female" (no neutral).
        model_path: Path to the SKEL model file or directory.
        simplify: Mesh simplification ratio. 1.0 = original mesh, 2.0 = half faces, etc.

    Note:
        forward_skeleton_mesh is NOT available in JAX backend.
        Use the PyTorch backend for skeleton mesh computation.
    """

    NUM_BETAS = 10
    NUM_JOINTS = 24
    NUM_POSE_PARAMS = 46

    def __init__(self, gender: str, model_path: Path | str | None = None, simplify: float = 1.0):
        assert gender in ("male", "female")
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"
        self.gender = gender

        skel_file = get_model_path(model_path, gender)
        with open(skel_file, "rb") as f:
            data = pkl.load(f)

        self._init_buffers(data, simplify)
        self._init_joints()

    def _init_buffers(self, data: dict, simplify: float) -> None:
        # Load full-resolution data
        v_template_full = np.asarray(data["skin_template_v"], dtype=np.float32)
        faces = np.asarray(data["skin_template_f"], dtype=np.int32)
        shapedirs_full = np.asarray(data["shapedirs"][:, :, : self.NUM_BETAS], dtype=np.float32)
        posedirs_full = np.asarray(data["posedirs"], dtype=np.float32)
        skin_weights = _sparse_to_dense(data["skin_weights"])

        # Apply mesh simplification if requested
        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            v_template, faces, vertex_map = simplify_mesh(v_template_full, faces, target_faces)
            shapedirs = shapedirs_full[vertex_map]
            posedirs = posedirs_full[vertex_map]
            skin_weights = skin_weights[vertex_map]
        else:
            v_template = v_template_full
            shapedirs = shapedirs_full
            posedirs = posedirs_full

        # Store as nnx.Variable
        self._v_template_full = nnx.Variable(jnp.asarray(v_template_full))
        self.shapedirs_full = nnx.Variable(jnp.asarray(shapedirs_full))
        self.posedirs_full = nnx.Variable(jnp.asarray(posedirs_full))
        self.J_regressor = nnx.Variable(jnp.asarray(_sparse_to_dense(data["J_regressor_osim"])))

        self._v_template = nnx.Variable(jnp.asarray(v_template))
        self._faces = nnx.Variable(jnp.asarray(faces))
        self.shapedirs = nnx.Variable(jnp.asarray(shapedirs))
        self.posedirs = nnx.Variable(jnp.asarray(posedirs))
        self._skin_weights = nnx.Variable(jnp.asarray(skin_weights))

        # Anatomical pose data
        apose = np.asarray(data["apose_rel_transfo"], dtype=np.float32)
        self.apose_R = nnx.Variable(jnp.asarray(apose[:, :3, :3]))
        self.apose_t = nnx.Variable(jnp.asarray(apose[:, :3, 3]))
        self.per_joint_rot = nnx.Variable(jnp.asarray(data["per_joint_rot"], dtype=np.float32))

        # Kinematic tree
        kintree = np.asarray(data["osim_kintree_table"], dtype=np.int64)
        id_to_col = {kintree[1, i]: i for i in range(kintree.shape[1])}
        self.parent_list = [id_to_col[kintree[0, i]] for i in range(1, kintree.shape[1])]
        self.parent = nnx.Variable(jnp.asarray(self.parent_list, dtype=jnp.int32))

        # Child indices for bone orientation
        child_list = []
        for i in range(self.NUM_JOINTS):
            children = np.where(kintree[0] == i)[0]
            child_list.append(kintree[1, children[0]] if len(children) > 0 else 0)
        self.child = nnx.Variable(jnp.asarray(child_list, dtype=jnp.int32))

        # Indices where bone orientation is fixed
        self.fixed_orientation_joints = nnx.Variable(jnp.asarray([0, 5, 10, 13, 18, 23], dtype=jnp.int32))

        # Number of SMPL joints
        self.num_joints_smpl = data["J_regressor"].shape[0]

        # Feet offset for Y=0 floor
        y_offset = -v_template[:, 1].min()
        self._feet_offset = nnx.Variable(jnp.asarray([0.0, y_offset, 0.0], dtype=jnp.float32))

    def _init_joints(self) -> None:
        """Precompute axes and indices for vectorized joint rotation computation."""
        from nanomanifold import SO3

        def pin_axis_from_euler(euler_xyz):
            """Compute pin joint axis from euler angles (XYZ convention)."""
            euler = np.array(euler_xyz, dtype=np.float32)
            R = SO3.to_matrix(SO3.from_euler(euler, convention="XYZ"))
            axis = R @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
            return axis.tolist()

        joint_axes = [
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],  # pelvis
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],  # femur_r
            [[0, 0, -1]],  # tibia_r
            [pin_axis_from_euler([0.175895, -0.105208, 0.0186622])],  # talus_r
            [pin_axis_from_euler([-1.76818999, 0.906223, 1.8196])],  # calcn_r
            [pin_axis_from_euler([-3.14158999, 0.619901, 0])],  # toes_r
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],  # femur_l
            [[0, 0, -1]],  # tibia_l
            [pin_axis_from_euler([0.175895, -0.105208, 0.0186622])],  # talus_l
            [pin_axis_from_euler([1.76818999, -0.906223, 1.8196])],  # calcn_l
            [pin_axis_from_euler([-3.14158999, -0.619901, 0])],  # toes_l
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],  # lumbar_body
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],  # thorax
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],  # head
            [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],  # scapula_r
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # humerus_r
            [[0.0494, 0.0366, 0.99810825]],  # ulna_r (MultiAxisJoint)
            [[-0.01716099, 0.99266564, -0.11966796]],  # radius_r (MultiAxisJoint)
            [[1, 0, 0], [0, 0, -1]],  # hand_r
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],  # scapula_l
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # humerus_l
            [[-0.0494, -0.0366, 0.99810825]],  # ulna_l (MultiAxisJoint)
            [[0.01716099, -0.99266564, -0.11966796]],  # radius_l (MultiAxisJoint)
            [[-1, 0, 0], [0, 0, -1]],  # hand_l
        ]

        all_axes: list[list[float]] = []
        rotation_indices: list[list[int]] = []
        dof_idx = 0

        for axes in joint_axes:
            n_dof = len(axes)

            for ax in axes:
                ax_arr = np.array(ax, dtype=np.float32)
                ax_norm = np.linalg.norm(ax_arr)
                if ax_norm > 1e-6 and abs(ax_norm - 1.0) > 1e-6:
                    ax_arr = ax_arr / ax_norm
                all_axes.append(ax_arr.tolist())

            identity_idx = self.NUM_POSE_PARAMS
            if n_dof == 1:
                rotation_indices.append([dof_idx, identity_idx, identity_idx])
            elif n_dof == 2:
                rotation_indices.append([dof_idx, dof_idx + 1, identity_idx])
            else:
                rotation_indices.append([dof_idx, dof_idx + 1, dof_idx + 2])

            dof_idx += n_dof

        all_axes.append([0.0, 0.0, 0.0])

        self._all_axes = nnx.Variable(jnp.asarray(all_axes, dtype=jnp.float32))
        self._rotation_indices = nnx.Variable(jnp.asarray(rotation_indices, dtype=jnp.int32))

        self._scapula_r_axes = nnx.Variable(jnp.asarray([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=jnp.float32))
        self._scapula_l_axes = nnx.Variable(jnp.asarray([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=jnp.float32))
        self._spine_axes = nnx.Variable(jnp.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=jnp.float32))

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self._faces[...]

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def num_vertices(self) -> int:
        return self._v_template[...].shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V 24"]:
        return self._skin_weights[...]

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self._v_template[...] + self._feet_offset[...]

    # -------------------------------------------------------------------------
    # Forward API
    # -------------------------------------------------------------------------

    def forward_vertices(
        self,
        shape: Float[jax.Array, "B|1 10"],
        pose: Float[jax.Array, "B 46"],
        global_rotation: Float[jax.Array, "B 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
    ) -> Float[jax.Array, "B V 3"]:
        """Compute mesh vertices [B, V, 3]."""
        return core.forward_vertices(
            v_template=self._v_template[...],
            v_template_full=self._v_template_full[...],
            shapedirs=self.shapedirs[...],
            shapedirs_full=self.shapedirs_full[...],
            posedirs=self.posedirs[...].reshape(-1, self.posedirs[...].shape[-1]).T,
            skin_weights=self._skin_weights[...],
            J_regressor=self.J_regressor[...],
            parents=self.parent[...],
            all_axes=self._all_axes[...],
            rotation_indices=self._rotation_indices[...],
            apose_R=self.apose_R[...],
            apose_t=self.apose_t[...],
            per_joint_rot=self.per_joint_rot[...],
            child=self.child[...],
            fixed_orientation_joints=self.fixed_orientation_joints[...],
            feet_offset=self._feet_offset[...],
            num_joints_smpl=self.num_joints_smpl,
            scapula_r_axes=self._scapula_r_axes[...],
            scapula_l_axes=self._scapula_l_axes[...],
            spine_axes=self._spine_axes[...],
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
        )

    def forward_skeleton(
        self,
        shape: Float[jax.Array, "B|1 10"],
        pose: Float[jax.Array, "B 46"],
        global_rotation: Float[jax.Array, "B 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
    ) -> Float[jax.Array, "B 24 4 4"]:
        """Compute skeleton joint transforms [B, 24, 4, 4]."""
        return core.forward_skeleton(
            v_template_full=self._v_template_full[...],
            shapedirs_full=self.shapedirs_full[...],
            J_regressor=self.J_regressor[...],
            parents=self.parent[...],
            all_axes=self._all_axes[...],
            rotation_indices=self._rotation_indices[...],
            apose_R=self.apose_R[...],
            apose_t=self.apose_t[...],
            per_joint_rot=self.per_joint_rot[...],
            child=self.child[...],
            fixed_orientation_joints=self.fixed_orientation_joints[...],
            feet_offset=self._feet_offset[...],
            scapula_r_axes=self._scapula_r_axes[...],
            scapula_l_axes=self._scapula_l_axes[...],
            spine_axes=self._spine_axes[...],
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jax.Array]:
        return {
            "shape": jnp.zeros((1, self.NUM_BETAS), dtype=dtype),
            "pose": jnp.zeros((batch_size, self.NUM_POSE_PARAMS), dtype=dtype),
            "global_rotation": jnp.zeros((batch_size, 3), dtype=dtype),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }


def _sparse_to_dense(arr_coo) -> np.ndarray:
    """Convert scipy sparse COO matrix to dense NumPy array."""
    if sparse.issparse(arr_coo):
        return np.asarray(arr_coo.toarray(), dtype=np.float32)
    return np.asarray(arr_coo, dtype=np.float32)


def from_native_args(
    shape: jax.Array,
    body_pose: jax.Array,
    root_rotation: jax.Array | None = None,
    global_rotation: jax.Array | None = None,
    global_translation: jax.Array | None = None,
) -> dict[str, jax.Array | None]:
    """Convert native SKEL args to forward_* kwargs."""
    return core.from_native_args(shape, body_pose, root_rotation, global_rotation, global_translation)


def to_native_outputs(
    vertices: jax.Array,
    transforms: jax.Array,
    feet_offset: jax.Array,
) -> dict[str, jax.Array]:
    """Convert forward_* outputs to native SKEL format."""
    return core.to_native_outputs(vertices, transforms, feet_offset)
