"""Shared engine for SMPL-derived linear blend skinning models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import _runtime as runtime_ops
from body_models._base import CorrectiveBasis, DenseCorrectiveBasis, PointRegressor, SkinnedModel
from body_models._common import deformation, kinematics, point_regression, skinning
from body_models._rotations import RotationType, rotation_ndim

Array = Any
RotationBlock: TypeAlias = tuple[Float[Array, "..."], RotationType]


class SmplFamilyModel(SkinnedModel):
    """Common state access and final deformation stages for the SMPL family."""

    NUM_SHAPE_COEFFS: int
    NUM_EXPR_COEFFS = 0
    _assets: Any
    rotation_type: RotationType

    @property
    def _num_rot_dims(self) -> int:
        return rotation_ndim(self.rotation_type)

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self._assets.faces

    @property
    def num_vertices(self) -> int:
        return self._assets.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[Array, "V J"]:
        return self._assets.lbs_weights

    @property
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self._assets.v_template

    @property
    def parents(self) -> list[int]:
        return list(self._assets.kinematic_tree.parents)

    @property
    def _corrective_basis(self) -> CorrectiveBasis:
        return DenseCorrectiveBasis(self._assets.posedirs)

    def prepare_point_regressor(
        self,
        mapping: Float[Array, "K V"],
    ) -> PointRegressor:
        """Preproject a vertex mapping and the family's linear identity bases."""
        regressor = super().prepare_point_regressor(mapping)
        xp = self._runtime.xp
        regressor["template"] = point_regression.project_vertex_values(
            regressor,
            self.rest_vertices,
            xp=xp,
        )
        regressor["identity_bases"] = tuple(
            point_regression.project_vertex_values(regressor, basis, xp=xp) for basis in self._point_identity_bases
        )
        return regressor

    @property
    def _point_identity_bases(self) -> tuple[Float[Array, "V 3 C"], ...]:
        bases = (self._assets.shapedirs,)
        return bases if self.NUM_EXPR_COEFFS == 0 else (*bases, self._assets.exprdirs)

    def _deform_linear_points(
        self,
        point_regressor: PointRegressor,
        identity_coefficients: Sequence[Float[Array, "*batch C"]],
        pose: deformation.SkinningPose,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
        global_translation: Float[Array, "*batch 3"] | None,
    ) -> Float[Array, "*batch K 3"]:
        xp = self._runtime.xp
        rest_points = point_regressor["template"]
        for coefficients, basis in zip(identity_coefficients, point_regressor["identity_bases"], strict=True):
            coefficient_dim = coefficients.shape[-1]
            if coefficient_dim > basis.shape[-1]:
                raise ValueError(f"identity coefficients exceed the model basis width of {basis.shape[-1]}")
            rest_points = rest_points + xp.einsum(
                "...c,kjdc->...kjd",
                coefficients,
                basis[..., :coefficient_dim],
            )
        points = point_regression.regress_points(point_regressor, rest_points, pose, xp=xp)
        return self._transform_points(points, point_regressor, global_rotation, global_translation)

    def _deform_vertices(
        self,
        identity: deformation.LinearIdentity,
        pose: deformation.SkinningPose,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
        global_translation: Float[Array, "*batch 3"] | None,
        vertex_indices: Sequence[int] | None,
    ) -> Float[Array, "*batch V 3"]:
        vertices = self._runtime._skin_vertices(
            self.apply_pose_correctives(identity=identity, pose=pose),
            pose["skinning_transforms"],
            skinning=self._assets.compact_skinning,
            vertex_indices=vertex_indices,
        )
        return skinning.apply_global_transform(
            vertices,
            global_rotation,
            global_translation,
            self.rotation_type,
            xp=self._runtime.xp,
        )

    def _transform_skeleton(
        self,
        skeleton: Float[Array, "*batch J 4 4"],
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
        global_translation: Float[Array, "*batch 3"] | None,
        joint_indices: Sequence[int] | None,
    ) -> Float[Array, "*batch selected 4 4"]:
        return skinning.transform_skeleton(
            skeleton,
            global_rotation,
            global_translation,
            self.rotation_type,
            joint_indices,
            xp=self._runtime.xp,
        )


def assemble_pose_matrices(
    blocks: Sequence[RotationBlock],
    root_rotation: Float[Array, "..."] | None,
    rotation_type: RotationType,
    *,
    xp: Any,
) -> Float[Array, "*batch J 3 3"]:
    """Convert ordered pose blocks to matrices and prepend the root rotation."""
    first_pose, first_type = blocks[0]
    first_pose_ndim = rotation_ndim(first_type) + 1
    batch_shape = tuple(first_pose.shape[:-first_pose_ndim])

    matrices = []
    for pose, source_type in blocks:
        pose_ndim = rotation_ndim(source_type) + 1
        if tuple(pose.shape[:-pose_ndim]) != batch_shape:
            raise ValueError("pose blocks must have the same batch shape")
        matrices.append(SO3.convert(pose, src=source_type, dst="rotmat", xp=xp))

    if root_rotation is None:
        root_matrices = SO3.identity_as(
            matrices[0],
            batch_dims=(*batch_shape, 1),
            rotation_type="rotmat",
            xp=xp,
        )
    else:
        root_matrices = SO3.convert(
            root_rotation,
            src=rotation_type,
            dst="rotmat",
            xp=xp,
        )[..., None, :, :]
    return xp.concat([root_matrices, *matrices], axis=-3)


def add_axis_angle_mean(
    pose: Float[Array, "..."],
    mean: Float[Array, "..."],
    rotation_type: RotationType,
    *,
    xp: Any,
) -> Float[Array, "*batch J 3"]:
    """Add a model's axis-angle mean to an encoded rotation block."""
    if rotation_type != "axis_angle":
        pose = SO3.convert(pose, src=rotation_type, dst="axis_angle", xp=xp)
    return pose + mean.reshape(-1, 3)


def forward_skeleton(
    runtime: runtime_ops.ArrayRuntime,
    tree: kinematics.KinematicTree,
    pose_matrices: Float[Array, "*batch J 3 3"],
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
) -> Float[Array, "*batch J 4 4"]:
    """Broadcast identity state and run family forward kinematics."""
    xp = runtime.xp
    batch_shape = tuple(pose_matrices.shape[:-3])
    offsets = xp.broadcast_to(local_joint_offsets, (*batch_shape, *local_joint_offsets.shape[-2:]))
    local_transforms = kinematics.affine_transforms(pose_matrices, offsets, xp=xp)
    return runtime._compose_kinematic_tree(local_transforms, tree)


def prepare_pose(
    runtime: runtime_ops.ArrayRuntime,
    tree: kinematics.KinematicTree,
    pose_matrices: Float[Array, "*batch J 3 3"],
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    rest_joints: Float[Array, "*identity_batch J 3"],
) -> deformation.SkinningPose:
    """Prepare transforms and pose offsets from assembled family rotations."""
    world_transforms = forward_skeleton(
        runtime,
        tree,
        pose_matrices,
        local_joint_offsets,
    )
    xp = runtime.xp
    batch_shape = tuple(pose_matrices.shape[:-3])
    rest_joints = xp.broadcast_to(rest_joints, (*batch_shape, *rest_joints.shape[-2:]))
    return {
        "skeleton_transforms": world_transforms,
        "skinning_transforms": skinning.bind_relative_transforms(world_transforms, rest_joints, xp=xp),
        "pose_coefficients": deformation.pose_coefficients(pose_matrices, xp=xp),
    }


def prepare_shape_identity(
    *,
    xp: Any,
    v_template: Float[Array, "V 3"],
    shapedirs: Float[Array, "V 3 S"],
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: Sequence[int],
    shape: Float[Array, "*batch S"],
) -> deformation.LinearIdentity:
    """Prepare vertex and joint identity state from shape coefficients."""
    shape_dim = shape.shape[-1]
    return deformation.prepare_linear_identity(
        vertex_template=v_template,
        vertex_directions=shapedirs[:, :, :shape_dim],
        joint_template=j_template,
        joint_directions=j_shapedirs[:, :, :shape_dim],
        parents=parents,
        coefficients=shape,
        xp=xp,
    )


def prepare_shape_skeleton_identity(
    *,
    xp: Any,
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: Sequence[int],
    shape: Float[Array, "*batch S"],
) -> deformation.SkeletonIdentity:
    """Prepare joint identity state from shape coefficients."""
    shape_dim = shape.shape[-1]
    return deformation.prepare_linear_skeleton(
        joint_template=j_template,
        joint_directions=j_shapedirs[:, :, :shape_dim],
        parents=parents,
        coefficients=shape,
        xp=xp,
    )


def prepare_shape_expression_identity(
    *,
    xp: Any,
    v_template: Float[Array, "V 3"],
    shapedirs: Float[Array, "V 3 S"],
    exprdirs: Float[Array, "V 3 E"],
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    j_exprdirs: Float[Array, "J 3 E"],
    parents: Sequence[int],
    shape: Float[Array, "*batch S"],
    expression: Float[Array, "*batch E"],
) -> deformation.LinearIdentity:
    """Prepare vertex and joint identity state from shape and expression."""
    coefficients, vertex_directions, joint_directions = _shape_expression_inputs(
        shape,
        expression,
        shapedirs,
        exprdirs,
        j_shapedirs,
        j_exprdirs,
        xp=xp,
    )
    return deformation.prepare_linear_identity(
        vertex_template=v_template,
        vertex_directions=vertex_directions,
        joint_template=j_template,
        joint_directions=joint_directions,
        parents=parents,
        coefficients=coefficients,
        xp=xp,
    )


def prepare_shape_expression_skeleton_identity(
    *,
    xp: Any,
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    j_exprdirs: Float[Array, "J 3 E"],
    parents: Sequence[int],
    shape: Float[Array, "*batch S"],
    expression: Float[Array, "*batch E"],
) -> deformation.SkeletonIdentity:
    """Prepare joint identity state from shape and expression."""
    _validate_coefficients(shape, expression)
    shape_dim = shape.shape[-1]
    expression_dim = expression.shape[-1]
    coefficients = xp.concat([shape, expression], axis=-1)
    joint_directions = xp.concat(
        [j_shapedirs[:, :, :shape_dim], j_exprdirs[:, :, :expression_dim]],
        axis=-1,
    )
    return deformation.prepare_linear_skeleton(
        joint_template=j_template,
        joint_directions=joint_directions,
        parents=parents,
        coefficients=coefficients,
        xp=xp,
    )


def _shape_expression_inputs(
    shape: Float[Array, "*batch S"],
    expression: Float[Array, "*batch E"],
    shapedirs: Float[Array, "V 3 S"],
    exprdirs: Float[Array, "V 3 E"],
    j_shapedirs: Float[Array, "J 3 S"],
    j_exprdirs: Float[Array, "J 3 E"],
    *,
    xp: Any,
) -> tuple[Float[Array, "*batch C"], Float[Array, "V 3 C"], Float[Array, "J 3 C"]]:
    _validate_coefficients(shape, expression)
    shape_dim = shape.shape[-1]
    expression_dim = expression.shape[-1]
    coefficients = xp.concat([shape, expression], axis=-1)
    vertex_directions = xp.concat(
        [shapedirs[:, :, :shape_dim], exprdirs[:, :, :expression_dim]],
        axis=-1,
    )
    joint_directions = xp.concat(
        [j_shapedirs[:, :, :shape_dim], j_exprdirs[:, :, :expression_dim]],
        axis=-1,
    )
    return coefficients, vertex_directions, joint_directions


def _validate_coefficients(
    shape: Float[Array, "*batch S"],
    expression: Float[Array, "*batch E"],
) -> None:
    if shape.ndim < 1 or shape.shape[-1] < 1:
        raise ValueError("shape must have shape [..., S] with S >= 1")
    if expression.ndim < 1 or expression.shape[-1] < 1:
        raise ValueError("expression must have shape [..., E] with E >= 1")


__all__ = [
    "SmplFamilyModel",
    "add_axis_angle_mean",
    "assemble_pose_matrices",
    "forward_skeleton",
    "prepare_pose",
    "prepare_shape_expression_identity",
    "prepare_shape_expression_skeleton_identity",
    "prepare_shape_identity",
    "prepare_shape_skeleton_identity",
]
