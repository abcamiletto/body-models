"""Backend-agnostic SOMA computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jaxtyping import Float, Int
from nanomanifold import SO3

from ...anny import core as anny_core
from ... import common
from ...common import get_namespace
from ...rotations import RotationType

Array = Any
Front = tuple[list[int], list[int]]


@dataclass(frozen=True)
class SomaCorrectives:
    corrective_bindpose: Any
    corrective_W1: Any
    corrective_W2_rows: Any
    corrective_W2_cols: Any
    corrective_W2_values: Any
    corrective_W2: Any


@dataclass(frozen=True)
class SomaTopology:
    parents_full: list[int]
    parents_full_index: Any
    joint_children_full: list[list[int]]
    joint_children_indices_full: Any
    skinned_vertex_indices_full: list[list[int]]
    skinned_vertex_indices_full_index: Any
    kinematic_fronts_full: list[Front]


@dataclass(frozen=True)
class SomaData:
    mean_full: Any
    mean_active: Any
    shapedirs_full: Any
    shapedirs_active: Any
    eigenvalues: Any
    bind_shape_full: Any
    bind_pose_world: Any
    bind_pose_local: Any
    t_pose_world: Any
    joint_regressor: Any
    skin_weights_full: Any
    skin_weights_active: Any
    faces: Any
    vertex_map: Any
    topology: SomaTopology
    correctives: SomaCorrectives

    def forward_vertices(self, *args, **kwargs):
        return _forward_vertices(self, *args, **kwargs)

    def forward_skeleton(self, *args, **kwargs):
        return forward_skeleton(self, *args, **kwargs)

    def prepare_identity_state(self, *args, **kwargs):
        return prepare_identity_state(self, *args, **kwargs)

    def apply_pose_correctives(self, *args, **kwargs):
        return apply_pose_correctives(self, *args, **kwargs)

    def linear_blend_skinning(self, *args, **kwargs):
        return linear_blend_skinning(*args, **kwargs)

    def repose_to_bind_pose(
        self,
        xp,
        rest_shape: Float[Array, "B V 3"],
        skin_weights: Float[Array, "V J"],
        world_bind_pose_fit: Float[Array, "B J 4 4"],
        bind_pose_local: Float[Array, "J 4 4"],
        kinematic_fronts: list[Front],
        parents_full_index: Int[Array, "J"],
    ) -> tuple[Float[Array, "B V 3"], Float[Array, "B J 4 4"]]:
        T_world = _repose_skeleton_to_bind_pose(
            xp=xp,
            world_bind_pose_fit=world_bind_pose_fit,
            bind_pose_local=bind_pose_local,
            kinematic_fronts=kinematic_fronts,
            parents_full_index=parents_full_index,
        )
        bone = T_world @ _invert_transforms(xp, world_bind_pose_fit)
        verts = self.linear_blend_skinning(xp, rest_shape, skin_weights, bone)
        return verts, T_world

    def pose_mesh_from_oriented_pose(
        self,
        xp,
        rest_shape: Float[Array, "B V 3"],
        skin_weights: Float[Array, "V Jf"],
        world_bind_pose: Float[Array, "B Jf 4 4"],
        kinematic_fronts: list[Front],
        parents_full_index: Int[Array, "Jf"],
        pose_rot_full: Float[Array, "B Jf 3 3"],
    ) -> tuple[Float[Array, "B V 3"], Float[Array, "B Jf 4 4"]]:
        T_world = _pose_skeleton_from_oriented_pose(
            xp=xp,
            world_bind_pose=world_bind_pose,
            kinematic_fronts=kinematic_fronts,
            parents_full_index=parents_full_index,
            pose_rot_full=pose_rot_full,
        )
        bone = T_world @ _invert_transforms(xp, world_bind_pose)
        verts = self.linear_blend_skinning(xp, rest_shape, skin_weights, bone)
        return verts, T_world


def _forward_vertices(
    data: SomaData,
    identity: Float[Array, "B|1 S"] | None,
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    rest_shape_full: Float[Array, "B|1 Vf 3"] | None = None,
    rest_shape_active: Float[Array, "B|1 Va 3"] | None = None,
    world_bind_pose_fit: Float[Array, "B|1 Jf 4 4"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    corrective_use_tanh: bool = True,
    apply_correctives: bool = True,
    rotation_type: RotationType = "axis_angle",
    match_warp: bool = True,
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3] in meters."""
    if xp is None:
        xp = get_namespace(identity if identity is not None else rest_shape_full)

    pose_rot = SO3.convert(pose, src=rotation_type, dst="rotmat", xp=xp)
    B = pose_rot.shape[0]
    topology = data.topology
    pose_rot_full = _orient_pose_rot_full(xp, pose_rot, data.t_pose_world, topology.parents_full_index)
    rest_shape_full, rest_shape_active, world_bind_pose_fit = _prepare_rest_shapes(
        xp=xp,
        batch_size=B,
        mean_full=data.mean_full,
        mean_active=data.mean_active,
        shapedirs_full=data.shapedirs_full,
        shapedirs_active=data.shapedirs_active,
        eigenvalues=data.eigenvalues,
        bind_shape=data.bind_shape_full,
        bind_pose_world=data.bind_pose_world,
        joint_regressor=data.joint_regressor,
        joint_children_full=topology.joint_children_full,
        joint_children_indices_full=topology.joint_children_indices_full,
        skinned_vertex_indices_full=topology.skinned_vertex_indices_full,
        skinned_vertex_indices_full_index=topology.skinned_vertex_indices_full_index,
        parents_full=topology.parents_full,
        identity=identity,
        rest_shape_full=rest_shape_full,
        rest_shape_active=rest_shape_active,
        world_bind_pose_fit=world_bind_pose_fit,
        match_warp=match_warp,
    )

    if apply_correctives:
        rest_shape_active, world_bind_pose = data.repose_to_bind_pose(
            xp=xp,
            rest_shape=rest_shape_active,
            skin_weights=data.skin_weights_active,
            world_bind_pose_fit=world_bind_pose_fit,
            bind_pose_local=data.bind_pose_local,
            kinematic_fronts=topology.kinematic_fronts_full,
            parents_full_index=topology.parents_full_index,
        )
        corrective_offsets = data.apply_pose_correctives(
            pose_rot_full=pose_rot_full,
            use_tanh=corrective_use_tanh,
            xp=xp,
        )
        if data.vertex_map is not None:
            corrective_offsets = corrective_offsets[:, data.vertex_map]
        rest_shape_active = rest_shape_active + corrective_offsets
    else:
        world_bind_pose = world_bind_pose_fit

    verts_cm, _ = data.pose_mesh_from_oriented_pose(
        xp=xp,
        rest_shape=rest_shape_active,
        skin_weights=data.skin_weights_active,
        world_bind_pose=world_bind_pose,
        kinematic_fronts=topology.kinematic_fronts_full,
        parents_full_index=topology.parents_full_index,
        pose_rot_full=pose_rot_full,
    )

    verts = verts_cm * 0.01
    verts = _apply_global_transform_vertices(xp, verts, global_rotation, global_translation, rotation_type)
    if vertex_indices is not None:
        verts = verts[:, xp.asarray(vertex_indices)]
    return verts


def forward_skeleton(
    data: SomaData,
    identity: Float[Array, "B|1 S"] | None,
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    rest_shape_full: Float[Array, "B|1 V 3"] | None = None,
    world_bind_pose_fit: Float[Array, "B|1 J 4 4"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    apply_correctives: bool = True,
    rotation_type: RotationType = "axis_angle",
    match_warp: bool = True,
    *,
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
    """Compute skeleton transforms [B, J, 4, 4] in meters."""
    if xp is None:
        xp = get_namespace(identity if identity is not None else rest_shape_full)

    pose_rot = SO3.convert(pose, src=rotation_type, dst="rotmat", xp=xp)
    B = pose_rot.shape[0]
    topology = data.topology
    rest_shape_full, world_bind_pose = _prepare_identity_from_inputs(
        xp=xp,
        batch_size=B,
        mean=data.mean_full,
        shapedirs=data.shapedirs_full,
        eigenvalues=data.eigenvalues,
        bind_shape=data.bind_shape_full,
        bind_pose_world=data.bind_pose_world,
        joint_regressor=data.joint_regressor,
        joint_children_full=topology.joint_children_full,
        joint_children_indices_full=topology.joint_children_indices_full,
        skinned_vertex_indices_full=topology.skinned_vertex_indices_full,
        skinned_vertex_indices_full_index=topology.skinned_vertex_indices_full_index,
        parents_full=topology.parents_full,
        identity=identity,
        rest_shape=rest_shape_full,
        world_bind_pose_fit=world_bind_pose_fit,
        match_warp=match_warp,
    )
    if apply_correctives:
        world_bind_pose = _repose_skeleton_to_bind_pose(
            xp=xp,
            world_bind_pose_fit=world_bind_pose,
            bind_pose_local=data.bind_pose_local,
            kinematic_fronts=topology.kinematic_fronts_full,
            parents_full_index=topology.parents_full_index,
        )
    pose_rot_full = _orient_pose_rot_full(xp, pose_rot, data.t_pose_world, topology.parents_full_index)
    T_world_cm = _pose_skeleton_from_oriented_pose(
        xp=xp,
        world_bind_pose=world_bind_pose,
        kinematic_fronts=topology.kinematic_fronts_full,
        parents_full_index=topology.parents_full_index,
        pose_rot_full=pose_rot_full,
    )

    T_world = common.set(T_world_cm, (..., slice(None, 3), 3), T_world_cm[..., :3, 3] * 0.01, xp=xp)
    T_world = _apply_global_transform_transforms(xp, T_world, global_rotation, global_translation, rotation_type)
    public = T_world[:, 1:]
    if joint_indices is not None:
        public = public[:, xp.asarray(joint_indices)]
    return public


def prepare_identity_state(
    data: SomaData,
    identity: Float[Array, "B|1 S"] | None,
    rest_shape_full: Float[Array, "B|1 Vf 3"] | None,
    rest_shape_active: Float[Array, "B|1 Va 3"] | None,
    match_warp: bool,
    *,
    xp: Any = None,
) -> tuple[Float[Array, "B Vf 3"], Float[Array, "B Va 3"], Float[Array, "B Jf 4 4"]]:
    if xp is None:
        xp = get_namespace(identity if identity is not None else rest_shape_full)

    identity_ref = identity if identity is not None else rest_shape_full
    if identity_ref is None:
        raise ValueError("SOMA identity preparation requires either identity or rest_shape_full.")

    return _prepare_rest_shapes(
        xp=xp,
        batch_size=identity_ref.shape[0],
        mean_full=data.mean_full,
        mean_active=data.mean_active,
        shapedirs_full=data.shapedirs_full,
        shapedirs_active=data.shapedirs_active,
        eigenvalues=data.eigenvalues,
        bind_shape=data.bind_shape_full,
        bind_pose_world=data.bind_pose_world,
        joint_regressor=data.joint_regressor,
        joint_children_full=data.topology.joint_children_full,
        joint_children_indices_full=data.topology.joint_children_indices_full,
        skinned_vertex_indices_full=data.topology.skinned_vertex_indices_full,
        skinned_vertex_indices_full_index=data.topology.skinned_vertex_indices_full_index,
        parents_full=data.topology.parents_full,
        identity=identity,
        rest_shape_full=rest_shape_full,
        rest_shape_active=rest_shape_active,
        world_bind_pose_fit=None,
        match_warp=match_warp,
    )


def _value(value):
    return value[...] if hasattr(value, "__getitem__") else value


def prepare_identity_shape(
    *,
    model_type: str,
    identity_model: Any,
    identity: Float[Array, "B I"],
    scale_params: Float[Array, "B K"] | None,
    num_scale_params: int | None,
    identity_internal_to_source_rotation: Float[Array, "3 3"],
    identity_internal_to_source_translation: Float[Array, "3"],
    identity_source_to_soma_rotation: Float[Array, "3 3"],
    identity_source_scale: float,
    identity_output_scale: float,
    identity_source_tetrahedra: Int[Array, "Fs 4"],
    identity_face_ids: Int[Array, "Vt"],
    identity_bary_coords: Float[Array, "Vt 4"],
    identity_unknown_ids: Int[Array, "U"],
    identity_anchor_ids: Int[Array, "A"],
    identity_solve_matrix: Float[Array, "U U"],
    identity_anchor_matrix: Float[Array, "U A"],
    identity_rhs_base: Float[Array, "U 3"],
    vertex_map: Int[Array, "Va"] | None,
    xp: Any,
) -> tuple[Float[Array, "B Vt 3"] | None, Float[Array, "B Va 3"] | None]:
    if model_type == "mhr":
        assert num_scale_params is not None
        rest_shape = mhr_identity_shape(
            model=identity_model,
            identity=identity,
            scale_params=scale_params,
            num_scale_params=num_scale_params,
            xp=xp,
        )
    elif model_type == "anny":
        anny_model = identity_model
        rest_shape = anny_identity_shape(
            template_vertices=_value(anny_model.template_vertices),
            blendshapes=_value(anny_model.blendshapes),
            phenotype_mask=_value(anny_model.phenotype_mask),
            anchors=anny_model._anchors if hasattr(anny_model, "_anchors") else anny_model._get_anchors_dict(),
            identity=identity,
            xp=xp,
        )
    else:
        linear_model = identity_model
        rest_shape = linear_identity_shape(
            mean=_value(linear_model.v_template_full),
            shapedirs=_value(linear_model.shapedirs_full),
            identity=identity,
            xp=xp,
        )

    rest_shape = apply_rigid_transform(
        rest_shape,
        rotation=_value(identity_internal_to_source_rotation),
        translation=_value(identity_internal_to_source_translation),
        xp=xp,
    )
    rest_shape = rest_shape * identity_source_scale
    rest_shape = transfer_identity_rest_shape(
        source_shape=rest_shape,
        source_tetrahedra=_value(identity_source_tetrahedra),
        face_ids=_value(identity_face_ids),
        bary_coords=_value(identity_bary_coords),
        unknown_ids=_value(identity_unknown_ids),
        anchor_ids=_value(identity_anchor_ids),
        solve_matrix=_value(identity_solve_matrix),
        anchor_matrix=_value(identity_anchor_matrix),
        rhs_base=_value(identity_rhs_base),
        xp=xp,
    )
    rest_shape = apply_rigid_transform(
        rest_shape,
        rotation=_value(identity_source_to_soma_rotation),
        xp=xp,
    )
    rest_shape = rest_shape * identity_output_scale
    vertex_map = None if vertex_map is None else _value(vertex_map)
    rest_shape_active = rest_shape if vertex_map is None else rest_shape[:, vertex_map]
    return rest_shape, rest_shape_active


def transfer_identity_rest_shape(
    source_shape: Float[Array, "B Vs 3"],
    source_tetrahedra: Int[Array, "Fs 4"],
    face_ids: Int[Array, "Vt"],
    bary_coords: Float[Array, "Vt 4"],
    unknown_ids: Int[Array, "U"],
    anchor_ids: Int[Array, "A"],
    solve_matrix: Float[Array, "U U"],
    anchor_matrix: Float[Array, "U A"],
    rhs_base: Float[Array, "U 3"],
    *,
    xp: Any = None,
) -> Float[Array, "B Vt 3"]:
    if xp is None:
        xp = get_namespace(source_shape)

    tetra_faces = source_tetrahedra[:, :3]
    f0 = source_shape[:, tetra_faces[:, 0]]
    f1 = source_shape[:, tetra_faces[:, 1]]
    f2 = source_shape[:, tetra_faces[:, 2]]
    fabricated = f0 + xp.linalg.cross(f1 - f0, f2 - f0)
    source_shape_tet = xp.concat([source_shape, fabricated], axis=1)

    tet_indices = source_tetrahedra[face_ids]
    v0 = source_shape_tet[:, tet_indices[:, 0]]
    v1 = source_shape_tet[:, tet_indices[:, 1]]
    v2 = source_shape_tet[:, tet_indices[:, 2]]
    v3 = source_shape_tet[:, tet_indices[:, 3]]
    bc = bary_coords[None]
    target_shape = v0 * bc[..., 0:1] + v1 * bc[..., 1:2] + v2 * bc[..., 2:3] + v3 * bc[..., 3:4]

    if unknown_ids.shape[0] == 0:
        return target_shape

    B = target_shape.shape[0]
    num_unknown = unknown_ids.shape[0]
    num_anchor = anchor_ids.shape[0]
    anchor_vertices = target_shape[:, anchor_ids]
    anchor_vertices = xp.reshape(anchor_vertices.swapaxes(0, 1), (num_anchor, B * 3))
    rhs = xp.broadcast_to(rhs_base[:, None, :], (num_unknown, B, 3)).reshape(num_unknown, B * 3)
    rhs = rhs - anchor_matrix @ anchor_vertices
    unknown_vertices = xp.linalg.solve(solve_matrix, rhs)
    unknown_vertices = xp.reshape(unknown_vertices, (num_unknown, B, 3)).swapaxes(0, 1)
    return common.set(target_shape, (slice(None), unknown_ids), unknown_vertices, xp=xp)


def linear_identity_shape(
    mean: Float[Array, "V 3"],
    shapedirs: Float[Array, "V 3 S"] | Float[Array, "V 3 I"],
    identity: Float[Array, "B I"],
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    if xp is None:
        xp = get_namespace(identity)
    identity_dim = identity.shape[1]
    return mean[None] + xp.einsum("bi,vci->bvc", identity, shapedirs[..., :identity_dim])


def mhr_identity_shape(
    model: Any,
    identity: Float[Array, "B I"],
    scale_params: Float[Array, "B K"] | None,
    num_scale_params: int,
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    if xp is None:
        xp = get_namespace(identity)

    batch_size = identity.shape[0]
    if scale_params is None:
        scale_params = common.zeros_as(identity, shape=(batch_size, num_scale_params), xp=xp)
    zero_pose = common.zeros_as(identity, shape=(batch_size, model.pose_dim), xp=xp)
    zero_pose = common.set(zero_pose, (slice(None), slice(-num_scale_params, None)), scale_params, xp=xp)
    expression = common.zeros_as(identity, shape=(batch_size, model.EXPR_DIM), xp=xp)
    return model.forward_vertices(shape=identity, pose=zero_pose, expression=expression)


def anny_identity_shape(
    template_vertices: Float[Array, "V 3"],
    blendshapes: Float[Array, "S V 3"],
    phenotype_mask: Float[Array, "S P"],
    anchors: dict[str, Float[Array, "A"]],
    identity: Float[Array, "B 6"],
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    if xp is None:
        xp = get_namespace(identity)
    return anny_core.identity_shape(
        template_vertices=template_vertices,
        blendshapes=blendshapes,
        phenotype_mask=phenotype_mask,
        anchors=anchors,
        identity=identity,
        xp=xp,
    )


def fit_rigid_transform(
    source_points: Float[Array, "V 3"],
    target_points: Float[Array, "V 3"],
    *,
    xp: Any = None,
) -> tuple[Float[Array, "3 3"], Float[Array, "3"]]:
    if xp is None:
        xp = get_namespace(source_points)

    source_center = xp.mean(source_points, axis=0)
    target_center = xp.mean(target_points, axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    covariance = source_centered.swapaxes(-2, -1) @ target_centered
    U, _S, Vh = xp.linalg.svd(covariance)
    reflection = common.eye_as(covariance, batch_dims=(), xp=xp)
    det = xp.linalg.det(Vh.swapaxes(-2, -1) @ U.swapaxes(-2, -1))
    reflection = common.set(reflection, (-1, -1), xp.where(det < 0, -1.0, 1.0), xp=xp)
    rotation = Vh.swapaxes(-2, -1) @ reflection @ U.swapaxes(-2, -1)
    translation = target_center - source_center @ rotation.swapaxes(-2, -1)
    return rotation, translation


def apply_rigid_transform(
    points: Float[Array, "B V 3"],
    rotation: Float[Array, "3 3"],
    translation: Float[Array, "3"] | None = None,
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    if xp is None:
        xp = get_namespace(points)
    points = points @ rotation.swapaxes(-2, -1)
    if translation is not None:
        points = points + translation
    return points


def resolve_identity_inputs(
    identity: Float[Array, "B|1 I"] | None,
    scale_params: Float[Array, "B|1 K"] | None,
    *,
    batch_size: int,
    identity_dim: int,
    num_scale_params: int | None,
    ref: Array,
    xp: Any = None,
) -> tuple[Float[Array, "B I"], Float[Array, "B K"] | None]:
    if xp is None:
        xp = get_namespace(ref)

    if identity is None:
        identity = common.zeros_as(ref, shape=(1, identity_dim), xp=xp)
    if identity.shape[0] == 1 and batch_size > 1:
        identity = xp.broadcast_to(identity, (batch_size, identity.shape[-1]))

    if num_scale_params is None:
        if scale_params is not None:
            raise ValueError("scale_params is only supported for SOMA model_type='mhr'.")
        return identity, None

    if scale_params is None:
        scale_params = common.zeros_as(ref, shape=(1, num_scale_params), xp=xp)
    if scale_params.shape[0] == 1 and batch_size > 1:
        scale_params = xp.broadcast_to(scale_params, (batch_size, scale_params.shape[-1]))
    return identity, scale_params


def apply_pose_correctives(
    data: SomaData,
    pose_rot_full: Float[Array, "B J 3 3"],
    use_tanh: bool,
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute SOMA pose correctives from oriented local rotations."""
    if xp is None:
        xp = get_namespace(pose_rot_full)

    correctives = data.correctives
    B = pose_rot_full.shape[0]
    x = correctives.corrective_bindpose.swapaxes(-2, -1)[None] @ pose_rot_full
    x = common.set(x, (slice(None), slice(None), 0, 0), x[:, :, 0, 0] - 1.0, copy=False, xp=xp)
    x = common.set(x, (slice(None), slice(None), 1, 1), x[:, :, 1, 1] - 1.0, copy=False, xp=xp)
    feat = x[:, :, :, :2].reshape(B, -1)

    z = feat @ correctives.corrective_W1
    z = xp.maximum(z, xp.asarray(0.0, dtype=feat.dtype))
    if use_tanh:
        z = xp.tanh(z)

    contrib = z[:, correctives.corrective_W2_rows] * correctives.corrective_W2_values[None]
    out = common.zeros_as(z, shape=(B, data.mean_full.shape[0] * 3), xp=xp)
    out = xp.asarray(out, copy=True)
    batch_index = xp.broadcast_to(xp.arange(B)[:, None], contrib.shape)
    xp.add.at(out, (batch_index, xp.broadcast_to(correctives.corrective_W2_cols[None], contrib.shape)), contrib)
    return out.reshape(B, data.mean_full.shape[0], 3)


def _broadcast_identity(identity: Array, *, batch_size: int, xp: Any) -> Array:
    if identity.shape[0] == 1 and batch_size > 1:
        return xp.broadcast_to(identity, (batch_size, identity.shape[-1]))
    return identity


def _broadcast_rest_shape(rest_shape: Array, *, batch_size: int, xp: Any) -> Array:
    if rest_shape.shape[0] == 1 and batch_size > 1:
        return xp.broadcast_to(rest_shape, (batch_size, *rest_shape.shape[1:]))
    return rest_shape


def _identity_to_rest_vertices(
    xp,
    mean: Float[Array, "V 3"],
    shapedirs: Float[Array, "S V 3"],
    eigenvalues: Float[Array, "S"],
    identity: Float[Array, "B S"],
) -> Float[Array, "B V 3"]:
    coeffs = identity * xp.sqrt(eigenvalues)[None]
    return mean[None] + xp.einsum("bs,svc->bvc", coeffs, shapedirs)


def _fit_rest_shape_to_bind_pose(
    xp,
    bind_shape: Float[Array, "V 3"],
    bind_pose_world: Float[Array, "J 4 4"],
    joint_regressor: Float[Array, "J V"],
    joint_children_full: list[list[int]],
    joint_children_indices_full: Int[Array, "J C"],
    skinned_vertex_indices_full: list[list[int]],
    skinned_vertex_indices_full_index: Int[Array, "J K"],
    parents_full: list[int],
    rest_shape: Float[Array, "B V 3"],
    match_warp: bool,
) -> tuple[Float[Array, "B V 3"], Float[Array, "B J 4 4"]]:
    joint_positions = xp.einsum("jv,bvc->bjc", joint_regressor, rest_shape)
    world_bind_pose = _fit_joint_rotations(
        xp=xp,
        bind_shape=bind_shape,
        bind_pose_world=bind_pose_world,
        joint_children_full=joint_children_full,
        joint_children_indices_full=joint_children_indices_full,
        skinned_vertex_indices_full=skinned_vertex_indices_full,
        skinned_vertex_indices_full_index=skinned_vertex_indices_full_index,
        parents_full=parents_full,
        joint_positions=joint_positions,
        target_shape=rest_shape,
        match_warp=match_warp,
    )
    return rest_shape, world_bind_pose


def _prepare_identity_from_inputs(
    xp,
    batch_size: int,
    mean: Float[Array, "V 3"],
    shapedirs: Float[Array, "S V 3"],
    eigenvalues: Float[Array, "S"],
    bind_shape: Float[Array, "V 3"],
    bind_pose_world: Float[Array, "J 4 4"],
    joint_regressor: Float[Array, "J V"],
    joint_children_full: list[list[int]],
    joint_children_indices_full: Int[Array, "J C"],
    skinned_vertex_indices_full: list[list[int]],
    skinned_vertex_indices_full_index: Int[Array, "J K"],
    parents_full: list[int],
    identity: Float[Array, "B|1 S"] | None,
    rest_shape: Float[Array, "B|1 V 3"] | None,
    world_bind_pose_fit: Float[Array, "B|1 J 4 4"] | None,
    match_warp: bool,
) -> tuple[Float[Array, "B V 3"], Float[Array, "B J 4 4"]]:
    if rest_shape is not None and world_bind_pose_fit is not None:
        return (
            _broadcast_rest_shape(rest_shape, batch_size=batch_size, xp=xp),
            _broadcast_rest_shape(world_bind_pose_fit, batch_size=batch_size, xp=xp),
        )

    if rest_shape is not None:
        rest_shape = _broadcast_rest_shape(rest_shape, batch_size=batch_size, xp=xp)
    else:
        if identity is None:
            raise ValueError("SOMA forward pass requires either identity or a precomputed rest_shape.")
        identity = _broadcast_identity(identity, batch_size=batch_size, xp=xp)
        rest_shape = _identity_to_rest_vertices(xp, mean, shapedirs, eigenvalues, identity)
    return _fit_rest_shape_to_bind_pose(
        xp=xp,
        bind_shape=bind_shape,
        bind_pose_world=bind_pose_world,
        joint_regressor=joint_regressor,
        joint_children_full=joint_children_full,
        joint_children_indices_full=joint_children_indices_full,
        skinned_vertex_indices_full=skinned_vertex_indices_full,
        skinned_vertex_indices_full_index=skinned_vertex_indices_full_index,
        parents_full=parents_full,
        rest_shape=rest_shape,
        match_warp=match_warp,
    )


def _prepare_rest_shapes(
    xp,
    batch_size: int,
    mean_full: Float[Array, "Vf 3"],
    mean_active: Float[Array, "Va 3"],
    shapedirs_full: Float[Array, "S Vf 3"],
    shapedirs_active: Float[Array, "S Va 3"],
    eigenvalues: Float[Array, "S"],
    bind_shape: Float[Array, "Vf 3"],
    bind_pose_world: Float[Array, "Jf 4 4"],
    joint_regressor: Float[Array, "Jf Vf"],
    joint_children_full: list[list[int]],
    joint_children_indices_full: Int[Array, "Jf C"],
    skinned_vertex_indices_full: list[list[int]],
    skinned_vertex_indices_full_index: Int[Array, "Jf K"],
    parents_full: list[int],
    identity: Float[Array, "B|1 S"] | None,
    rest_shape_full: Float[Array, "B|1 Vf 3"] | None,
    rest_shape_active: Float[Array, "B|1 Va 3"] | None,
    world_bind_pose_fit: Float[Array, "B|1 Jf 4 4"] | None,
    match_warp: bool,
) -> tuple[Float[Array, "B Vf 3"], Float[Array, "B Va 3"], Float[Array, "B Jf 4 4"]]:
    full_shape, world_bind_pose = _prepare_identity_from_inputs(
        xp=xp,
        batch_size=batch_size,
        mean=mean_full,
        shapedirs=shapedirs_full,
        eigenvalues=eigenvalues,
        bind_shape=bind_shape,
        bind_pose_world=bind_pose_world,
        joint_regressor=joint_regressor,
        joint_children_full=joint_children_full,
        joint_children_indices_full=joint_children_indices_full,
        skinned_vertex_indices_full=skinned_vertex_indices_full,
        skinned_vertex_indices_full_index=skinned_vertex_indices_full_index,
        parents_full=parents_full,
        identity=identity,
        rest_shape=rest_shape_full,
        world_bind_pose_fit=world_bind_pose_fit,
        match_warp=match_warp,
    )
    if rest_shape_active is not None:
        active_shape = _broadcast_rest_shape(rest_shape_active, batch_size=batch_size, xp=xp)
    else:
        if identity is None:
            active_shape = full_shape
        else:
            identity = _broadcast_identity(identity, batch_size=batch_size, xp=xp)
            active_shape = _identity_to_rest_vertices(xp, mean_active, shapedirs_active, eigenvalues, identity)
    return full_shape, active_shape, world_bind_pose


def _repose_skeleton_to_bind_pose(
    xp,
    world_bind_pose_fit: Float[Array, "B J 4 4"],
    bind_pose_local: Float[Array, "J 4 4"],
    kinematic_fronts: list[Front],
    parents_full_index: Int[Array, "J"],
) -> Float[Array, "B J 4 4"]:
    B = world_bind_pose_fit.shape[0]
    bind_local_fit = _joint_world_to_local(xp, world_bind_pose_fit, parents_full_index)
    local_t = bind_local_fit[..., :3, 3]

    zeros = xp.asarray(0.0, dtype=local_t.dtype)
    local_t = common.set(local_t, (slice(None), 1, 0), zeros, copy=False, xp=xp)
    local_t = common.set(local_t, (slice(None), 1, 2), zeros, copy=False, xp=xp)

    bind_rot = xp.broadcast_to(bind_pose_local[None, :, :3, :3], (B, bind_pose_local.shape[0], 3, 3))
    T_local = _build_transform_matrix(xp, bind_rot, local_t)
    T_world = _forward_kinematics(xp, T_local, kinematic_fronts)

    y_shift = xp.amin(T_world[:, :, 1, 3], axis=1)
    return common.set(
        T_world,
        (slice(None), slice(None), 1, 3),
        T_world[:, :, 1, 3] - y_shift[:, None],
        xp=xp,
    )


def _orient_pose_rot_full(
    xp,
    pose_rot: Float[Array, "B J 3 3"],
    t_pose_world: Float[Array, "Jf 4 4"],
    parents_full_index: Int[Array, "Jf"],
) -> Float[Array, "B Jf 3 3"]:
    B = pose_rot.shape[0]
    root_identity = common.eye_as(pose_rot, batch_dims=(B, 1), xp=xp)
    pose_rot_full = xp.concat([root_identity, pose_rot], axis=1)
    orient = t_pose_world[:, :3, :3]
    orient_parent_T = orient[parents_full_index].swapaxes(-2, -1)
    return orient_parent_T[None] @ pose_rot_full @ orient[None]


def _pose_skeleton_from_oriented_pose(
    xp,
    world_bind_pose: Float[Array, "B Jf 4 4"],
    kinematic_fronts: list[Front],
    parents_full_index: Int[Array, "Jf"],
    pose_rot_full: Float[Array, "B Jf 3 3"],
) -> Float[Array, "B Jf 4 4"]:
    B = pose_rot_full.shape[0]
    bind_local = _joint_world_to_local(xp, world_bind_pose, parents_full_index)
    local_t = bind_local[..., :3, 3]
    if local_t.shape[1] > 1:
        zeros = common.zeros_as(local_t[:, 1], shape=(B, 3), xp=xp)
        local_t = common.set(local_t, (slice(None), 1), zeros, copy=False, xp=xp)

    T_local = _build_transform_matrix(xp, pose_rot_full, local_t)
    return _forward_kinematics(xp, T_local, kinematic_fronts)


def _fit_joint_rotations(
    xp,
    bind_shape: Float[Array, "V 3"],
    bind_pose_world: Float[Array, "J 4 4"],
    joint_children_full: list[list[int]],
    joint_children_indices_full: Int[Array, "J C"],
    skinned_vertex_indices_full: list[list[int]],
    skinned_vertex_indices_full_index: Int[Array, "J K"],
    parents_full: list[int],
    joint_positions: Float[Array, "B J 3"],
    target_shape: Float[Array, "B V 3"],
    match_warp: bool,
) -> Float[Array, "B J 4 4"]:
    B, J = joint_positions.shape[:2]
    bind_rot = bind_pose_world[:, :3, :3]
    bind_pos = bind_pose_world[:, :3, 3]

    rotations = [xp.broadcast_to(bind_rot[None, 0], (B, 3, 3))]
    for joint_index in range(1, J):
        children = joint_children_full[joint_index]
        if not children:
            parent_index = parents_full[joint_index]
            rotations.append(rotations[parent_index])
            continue

        skinned_vids = skinned_vertex_indices_full[joint_index]
        if skinned_vids:
            skinned_idx = skinned_vertex_indices_full_index[joint_index, : len(skinned_vids)]
            skinned_orig = bind_shape[skinned_idx] - bind_pos[joint_index]
            skinned_new = target_shape[:, skinned_idx, :] - joint_positions[:, joint_index : joint_index + 1, :]
            R_init = _align_vectors(
                xp,
                skinned_new,
                skinned_orig[None],
                match_warp=match_warp,
            )
        else:
            R_init = common.eye_as(bind_rot, batch_dims=(B,), xp=xp)

        child_idx = joint_children_indices_full[joint_index, : len(children)]
        pos_children_orig = bind_pos[child_idx] - bind_pos[joint_index : joint_index + 1]
        pos_children_orig = xp.einsum("bij,cj->bci", R_init, pos_children_orig)
        pos_children_new = joint_positions[:, child_idx, :] - joint_positions[:, joint_index : joint_index + 1, :]
        align_rot = _align_vectors(
            xp,
            pos_children_new,
            pos_children_orig,
            match_warp=match_warp,
        )
        R_joint = align_rot @ R_init @ bind_rot[None, joint_index]
        rotations.append(R_joint)

    R = xp.stack(rotations, axis=1)
    return _build_transform_matrix(xp, R, joint_positions)


def _align_vectors(
    xp,
    target: Float[Array, "B N 3"],
    source: Float[Array, "B N 3"],
    *,
    match_warp: bool,
) -> Float[Array, "B 3 3"]:
    if target.shape[-2] == 1:
        return _rotation_between_vectors(xp, target[:, 0], source[:, 0])

    H = xp.einsum("bni,bnj->bij", target, source)
    # SOMALayer's default warp path uses plain covariance; the PyTorch fallback adds a virtual normal.
    if not match_warp:
        p0, p1 = target[:, 0], target[:, 1]
        q0, q1 = source[:, 0], source[:, 1]
        n_target = xp.linalg.cross(p0, p1)
        n_source = xp.linalg.cross(q0, q1)
        len_target = xp.linalg.vector_norm(n_target, axis=-1, keepdims=True)
        len_source = xp.linalg.vector_norm(n_source, axis=-1, keepdims=True)
        scale_target = xp.linalg.vector_norm(p0, axis=-1, keepdims=True) / (len_target + 1e-8)
        scale_source = xp.linalg.vector_norm(q0, axis=-1, keepdims=True) / (len_source + 1e-8)
        valid = (len_target[:, 0] > 1e-9) & (len_source[:, 0] > 1e-9)
        v_target = n_target * scale_target
        v_source = n_source * scale_source
        virtual = xp.einsum("bi,bj->bij", v_target, v_source)
        H = xp.where(valid[:, None, None], H + virtual, H)
    return _kabsch(xp, H)


def _kabsch(xp, H: Float[Array, "B 3 3"]) -> Float[Array, "B 3 3"]:
    U, _, Vh = xp.linalg.svd(H)
    UVt = U @ Vh.swapaxes(-2, -1)
    det_sign = xp.where(_det3(UVt) < 0, xp.asarray(-1.0, dtype=H.dtype), xp.asarray(1.0, dtype=H.dtype))
    D = common.eye_as(H, batch_dims=(H.shape[0],), xp=xp)
    D = common.set(D, (slice(None), 2, 2), det_sign, xp=xp)
    return U @ D @ Vh


def _rotation_between_vectors(
    xp,
    target: Float[Array, "B 3"],
    source: Float[Array, "B 3"],
) -> Float[Array, "B 3 3"]:
    target_norm = xp.linalg.vector_norm(target, axis=-1, keepdims=True)
    source_norm = xp.linalg.vector_norm(source, axis=-1, keepdims=True)
    target_unit = target / xp.where(target_norm > 1e-8, target_norm, xp.ones_like(target_norm))
    source_unit = source / xp.where(source_norm > 1e-8, source_norm, xp.ones_like(source_norm))

    dot = xp.sum(target_unit * source_unit, axis=-1, keepdims=True)
    dot = xp.clip(dot, -1.0, 1.0)
    cross = xp.linalg.cross(source_unit, target_unit)
    cross_norm = xp.linalg.vector_norm(cross, axis=-1, keepdims=True)
    axis = cross / xp.where(cross_norm > 1e-8, cross_norm, xp.ones_like(cross_norm))

    antiparallel = dot[:, 0] < -1.0 + 1e-6
    basis = common.eye_as(target, batch_dims=(target.shape[0],), xp=xp)
    x_vec = basis[:, 0]
    y_vec = basis[:, 1]
    w = xp.where(xp.abs(source_unit[:, 0:1]) > 0.6, y_vec, x_vec)
    antiparallel_axis = xp.linalg.cross(source_unit, w)
    antiparallel_axis_norm = xp.linalg.vector_norm(antiparallel_axis, axis=-1, keepdims=True)
    antiparallel_axis = antiparallel_axis / xp.where(
        antiparallel_axis_norm > 1e-8,
        antiparallel_axis_norm,
        xp.ones_like(antiparallel_axis_norm),
    )
    axis = xp.where(antiparallel[:, None], antiparallel_axis, axis)

    angle = xp.atan2(cross_norm[:, 0], dot[:, 0])
    pi = xp.asarray(3.141592653589793, dtype=target.dtype)
    angle = xp.where(antiparallel, pi, angle)
    return SO3.conversions.from_axis_angle_to_rotmat(angle[..., None] * axis, xp=xp)


def _det3(M: Float[Array, "B 3 3"]) -> Float[Array, "B"]:
    return (
        M[:, 0, 0] * (M[:, 1, 1] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 1])
        - M[:, 0, 1] * (M[:, 1, 0] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 0])
        + M[:, 0, 2] * (M[:, 1, 0] * M[:, 2, 1] - M[:, 1, 1] * M[:, 2, 0])
    )


def _joint_world_to_local(
    xp, world: Float[Array, "B J 4 4"], parents_full_index: Int[Array, "J"]
) -> Float[Array, "B J 4 4"]:
    inv = _invert_transforms(xp, world)
    return inv[:, parents_full_index] @ world


def linear_blend_skinning(
    xp,
    bind_shape: Float[Array, "B V 3"],
    skin_weights: Float[Array, "V J"],
    bone_transforms: Float[Array, "B J 4 4"],
) -> Float[Array, "B V 3"]:
    R = bone_transforms[..., :3, :3]
    t = bone_transforms[..., :3, 3]
    R_blend = xp.einsum("vj,bjik->bvik", skin_weights, R)
    t_blend = xp.einsum("vj,bji->bvi", skin_weights, t)
    return xp.einsum("bvik,bvk->bvi", R_blend, bind_shape) + t_blend


def _invert_transforms(xp, T: Float[Array, "B J 4 4"]) -> Float[Array, "B J 4 4"]:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_T = R.swapaxes(-2, -1)
    t_inv = -xp.squeeze(R_T @ t[..., None], axis=-1)
    return _build_transform_matrix(xp, R_T, t_inv)


def _build_transform_matrix(
    xp,
    R: Float[Array, "... 3 3"],
    t: Float[Array, "... 3"],
) -> Float[Array, "... 4 4"]:
    T = common.zeros_as(R, shape=(*R.shape[:-2], 4, 4), xp=xp)
    T = common.set(T, (..., slice(None, 3), slice(None, 3)), R, xp=xp)
    T = common.set(T, (..., slice(None, 3), 3), t, xp=xp)
    T = common.set(T, (..., 3, 3), xp.asarray(1.0, dtype=R.dtype), xp=xp)
    return T


def _forward_kinematics(xp, T_local: Float[Array, "B J 4 4"], fronts: list[Front]) -> Float[Array, "B J 4 4"]:
    J = T_local.shape[1]
    world: list[Float[Array, "B 4 4"] | None] = [None] * J
    for joints, parents in fronts:
        if parents[0] < 0:
            for joint in joints:
                world[joint] = T_local[:, joint]
            continue
        for joint, parent in zip(joints, parents):
            world[joint] = world[parent] @ T_local[:, joint]
    return xp.stack(world, axis=1)


def _apply_global_transform_vertices(
    xp,
    vertices: Float[Array, "B V 3"],
    rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None,
    translation: Float[Array, "B 3"] | None,
    rotation_type: RotationType,
) -> Float[Array, "B V 3"]:
    if rotation is not None:
        R = SO3.convert(rotation, src=rotation_type, dst="rotmat", xp=xp)
        vertices = xp.einsum("bij,bvj->bvi", R, vertices)
    if translation is not None:
        vertices = vertices + translation[:, None]
    return vertices


def _apply_global_transform_transforms(
    xp,
    transforms: Float[Array, "B J 4 4"],
    rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None,
    translation: Float[Array, "B 3"] | None,
    rotation_type: RotationType,
) -> Float[Array, "B J 4 4"]:
    if rotation is None and translation is None:
        return transforms

    B = transforms.shape[0]
    global_T = common.zeros_as(transforms, shape=(B, 4, 4), xp=xp)
    global_T = common.set(global_T, (slice(None), 3, 3), xp.asarray(1.0, dtype=transforms.dtype), xp=xp)
    if rotation is not None:
        R = SO3.convert(rotation, src=rotation_type, dst="rotmat", xp=xp)
        global_T = common.set(global_T, (slice(None), slice(None, 3), slice(None, 3)), R, xp=xp)
    else:
        eye = common.eye_as(transforms, batch_dims=(B,), xp=xp)
        global_T = common.set(global_T, (slice(None), slice(None, 3), slice(None, 3)), eye[:, :3, :3], xp=xp)
    if translation is not None:
        global_T = common.set(global_T, (slice(None), slice(None, 3), 3), translation, xp=xp)
    return xp.einsum("bij,bnjk->bnik", global_T, transforms)


@dataclass(frozen=True)
class SomaOps:
    prepare_identity_shape: Any = prepare_identity_shape
    fit_rigid_transform: Any = fit_rigid_transform
    resolve_identity_inputs: Any = resolve_identity_inputs


ops = SomaOps()
