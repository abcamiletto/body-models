"""Shared engine for linear blendshape models with articulated skinning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import _runtime as runtime_ops
from body_models._base import CorrectiveBasis, DenseCorrectiveBasis, PointRegressor, SkinnedModel
from body_models._common import deformation, kinematics, point_regression, skinning
from body_models._rotations import RotationType, rotation_ndim

Array = Any


@dataclass(frozen=True)
class PoseBlock:
    """One contiguous run of joint rotations and its optional axis-angle mean."""

    pose: Float[Array, "..."]
    rotation_type: RotationType
    axis_angle_mean: Float[Array, "..."] | None = None


@dataclass(frozen=True)
class IdentityBasis:
    """Directions driven by one identity coefficient vector, such as ``shape``."""

    vertices: Float[Array, "V 3 C"]
    joints: Float[Array, "J 3 C"] | None
    """Joint directions, or None when the coefficients do not move the skeleton."""


class LinearBlendshapeModel(SkinnedModel):
    """Common state access and deformation stages for linear blendshape models."""

    _assets: Any
    rotation_type: RotationType

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

    @property
    def _identity_bases(self) -> tuple[IdentityBasis, ...]:
        """Bases in the order of the model's identity parameters."""
        return (IdentityBasis(self._assets.shapedirs, self._assets.j_shapedirs),)

    def prepare_point_regressor(
        self,
        mapping: Float[Array, "K V"],
    ) -> PointRegressor:
        """Preproject a vertex mapping and the model's linear identity bases."""
        regressor = super().prepare_point_regressor(mapping)
        xp = self._runtime.xp
        regressor["template"] = point_regression.project_vertex_values(regressor, self.rest_vertices, xp=xp)
        regressor["identity_bases"] = tuple(
            point_regression.project_vertex_values(regressor, basis.vertices, xp=xp) for basis in self._identity_bases
        )
        return regressor

    def _prepare_linear_identity(self, *coefficients: Float[Array, "*batch C"]) -> deformation.LinearIdentity:
        skeleton = self._prepare_skeleton_identity(*coefficients)
        rest_vertices = self._assets.v_template
        for basis, values in zip(self._identity_bases, coefficients, strict=True):
            directions = basis.vertices[..., : values.shape[-1]]
            rest_vertices = deformation.blend_shapes(rest_vertices, directions, values, xp=self._runtime.xp)
        return {
            "coefficients": skeleton["coefficients"],
            "rest_joints": skeleton["rest_joints"],
            "local_joint_offsets": skeleton["local_joint_offsets"],
            "rest_vertices": rest_vertices,
        }

    def _prepare_skeleton_identity(self, *coefficients: Float[Array, "*batch C"]) -> deformation.LinearSkeletonIdentity:
        xp = self._runtime.xp
        rest_joints = self._assets.j_template
        for basis, values in zip(self._identity_bases, coefficients, strict=True):
            if basis.joints is not None:
                rest_joints = deformation.blend_shapes(
                    rest_joints, basis.joints[..., : values.shape[-1]], values, xp=xp
                )
        parents = self._assets.kinematic_tree.parents
        return {
            "coefficients": coefficients,
            "rest_joints": rest_joints,
            "local_joint_offsets": kinematics.local_joint_offsets(rest_joints, parents, xp=xp),
        }

    def _assemble_pose_matrices(
        self,
        blocks: Sequence[PoseBlock],
        root_rotation: Float[Array, "..."] | None,
        selection: kinematics.JointSelection | None,
    ) -> Float[Array, "*batch J 3 3"]:
        return assemble_pose_matrices(self._runtime, blocks, root_rotation, self.rotation_type, selection)

    def _prepare_linear_pose(
        self,
        pose_matrices: Float[Array, "*batch J 3 3"],
        identity: deformation.LinearSkeletonIdentity,
    ) -> deformation.SkinningPose:
        """Prepare transforms and pose-corrective coefficients from assembled rotations."""
        xp = self._runtime.xp
        world_transforms = self._posed_joints(pose_matrices, identity, None)
        batch_shape = tuple(pose_matrices.shape[:-3])
        rest_joints = xp.broadcast_to(identity["rest_joints"], (*batch_shape, *identity["rest_joints"].shape[-2:]))
        return {
            "skeleton_transforms": world_transforms,
            "skinning_transforms": skinning.bind_relative_transforms(world_transforms, rest_joints, xp=xp),
            "pose_coefficients": deformation.pose_coefficients(pose_matrices, xp=xp),
        }

    def _joint_selection(self, joint_indices: Sequence[int] | None) -> kinematics.JointSelection | None:
        return None if joint_indices is None else self._assets.kinematic_tree.select(joint_indices)

    def _posed_joints(
        self,
        pose_matrices: Float[Array, "*batch J 3 3"],
        identity: deformation.LinearSkeletonIdentity,
        selection: kinematics.JointSelection | None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Run forward kinematics, restricted to ``selection`` when given."""
        xp = self._runtime.xp
        runtime = self._runtime
        batch_shape = tuple(pose_matrices.shape[:-3])
        offsets = identity["local_joint_offsets"]
        offsets = xp.broadcast_to(offsets, (*batch_shape, *offsets.shape[-2:]))
        tree = self._assets.kinematic_tree
        if selection is not None:
            offsets = offsets[..., runtime.asarray(selection.cover_indices, like=offsets, dtype=xp.int32), :]
            tree = selection.tree
        world_transforms = runtime._compose_kinematic_tree(
            kinematics.affine_transforms(pose_matrices, offsets, xp=xp), tree
        )
        if selection is None:
            return world_transforms
        output_indices = runtime.asarray(selection.output_indices, like=world_transforms, dtype=xp.int32)
        return world_transforms[..., output_indices, :, :]

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

    def _deform_linear_points(
        self,
        point_regressor: PointRegressor,
        identity: deformation.LinearSkeletonIdentity,
        pose: deformation.SkinningPose,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
        global_translation: Float[Array, "*batch 3"] | None,
    ) -> Float[Array, "*batch K 3"]:
        """Regress points from projected identity bases, without building rest vertices."""
        xp = self._runtime.xp
        rest_points = point_regressor["template"]
        for values, basis in zip(identity["coefficients"], point_regressor["identity_bases"], strict=True):
            rest_points = rest_points + xp.einsum("...c,kjdc->...kjd", values, basis[..., : values.shape[-1]])
        points = point_regression.regress_points(point_regressor, rest_points, pose, xp=xp)
        return self._transform_points(points, point_regressor, global_rotation, global_translation)

    def _transform_skeleton(
        self,
        skeleton: Float[Array, "*batch J 4 4"],
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
        global_translation: Float[Array, "*batch 3"] | None,
    ) -> Float[Array, "*batch selected 4 4"]:
        return skinning.transform_skeleton(
            skeleton,
            global_rotation,
            global_translation,
            self.rotation_type,
            xp=self._runtime.xp,
        )


def assemble_pose_matrices(
    runtime: runtime_ops.ArrayRuntime,
    blocks: Sequence[PoseBlock],
    root_rotation: Float[Array, "..."] | None,
    rotation_type: RotationType,
    selection: kinematics.JointSelection | None = None,
) -> Float[Array, "*batch J 3 3"]:
    """Convert ordered pose blocks to matrices and prepend the root rotation."""
    xp = runtime.xp
    first = blocks[0]
    first_pose_ndim = rotation_ndim(first.rotation_type) + 1
    batch_shape = tuple(first.pose.shape[:-first_pose_ndim])
    pose_positions = None
    if selection is not None:
        pose_positions = tuple(joint - 1 for joint in selection.cover_indices if joint > 0)

    matrices = []
    offset = 0
    for block in blocks:
        pose_ndim = rotation_ndim(block.rotation_type) + 1
        if tuple(block.pose.shape[:-pose_ndim]) != batch_shape:
            raise ValueError("pose blocks must have the same batch shape")

        pose = block.pose
        mean = block.axis_angle_mean
        num_joints = pose.shape[-pose_ndim]
        if pose_positions is not None:
            block_positions = tuple(
                position - offset for position in pose_positions if offset <= position < offset + num_joints
            )
            selector = runtime.asarray(block_positions, like=pose, dtype=xp.int32)
            if rotation_ndim(block.rotation_type) > 1:
                pose = pose[..., selector, :, :]
            else:
                pose = pose[..., selector, :]
            if mean is not None:
                mean = mean.reshape(-1, 3)[selector]
        offset += num_joints

        source_type = block.rotation_type
        if mean is not None:
            if source_type != "axis_angle":
                pose = SO3.convert(pose, src=source_type, dst="axis_angle", xp=xp)
            pose = pose + mean.reshape(-1, 3)
            source_type = "axis_angle"
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
    if selection is not None and not selection.cover_indices:
        return xp.concat(matrices, axis=-3)
    return xp.concat([root_matrices, *matrices], axis=-3)


__all__ = ["IdentityBasis", "LinearBlendshapeModel", "PoseBlock", "assemble_pose_matrices"]
