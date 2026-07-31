"""Generate an authored, shape-matched, rigid SMPL character."""

from __future__ import annotations

import json
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from scipy.spatial import transform as scipy_transform
from trimesh import Trimesh

from body_models.smpl import SMPL
from body_models.smpl_humanoid._constants import (
    BODY_JOINTS,
    FINGER_JOINT_NAMES,
    JOINT_NAMES,
    PARENTS,
    ROBOT_JOINT_NAMES,
    ROBOT_PARENTS,
)
from body_models.smplx import SMPLX

_SOURCE_ASSET = "assets/smpl_robot_professional.glb"
_ARMOR = "0.68 0.48 0.24 1"
_JOINT = "0.52 0.34 0.16 1"
_BODY_JOINTS = {name for name, _ in BODY_JOINTS}
_ACTUATED_JOINTS = _BODY_JOINTS | set(FINGER_JOINT_NAMES)
_AUTHORED_NAME = re.compile(r"^J(?P<joint>\d{2})__.*__(?P<material>armor|joint)$")
_BILATERAL_JOINT_PAIRS = (
    (1, 2),
    (4, 5),
    (7, 8),
    (10, 11),
    (13, 14),
    (16, 17),
    (18, 19),
    (20, 21),
    (22, 23),
    *((index, index + 15) for index in range(24, 39)),
)
_SAGITTAL_REFLECTION = np.array([-1.0, 1.0, 1.0])
_PRESERVED_MESH_FRAMES = {
    "Neck": "Neck",
    "Head": "Neck",
    "L_Hand": "L_Wrist",
    "R_Hand": "R_Wrist",
}
_BLENDER_TO_MODEL = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)


@dataclass(frozen=True)
class _SourceGeometry:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    color: str
    collision: bool
    joint_index: int


@dataclass(frozen=True)
class _RigidPart:
    name: str
    joint_index: int
    vertices: np.ndarray
    faces: np.ndarray
    color: str
    collision: bool
    shoulder_socket_weights: np.ndarray | None = None

    @property
    def mesh(self) -> Trimesh:
        return Trimesh(vertices=self.vertices, faces=self.faces, process=False)


def generate_smpl_robot(
    output_path: Path | str,
    *,
    shape: np.ndarray,
    model_path: Path | str | None = None,
    gender: Literal["neutral", "male", "female"] | None = None,
    source_model: SMPL | None = None,
    smplx_model: SMPLX | None = None,
) -> Path:
    """Generate one standalone, non-skinned MJCF character for an SMPL shape.

    The bundled authored character is hard-partitioned into 54 rigid link
    meshes. SMPL shape changes only measured bone lengths: neutral bone
    directions and all transverse shell dimensions remain fixed. The exported
    MJCF has no skin weights, blend shapes, or runtime dependency on SMPL.

    Args:
        output_path: Destination MJCF XML file.
        shape: One to ten SMPL shape coefficients.
        model_path: Licensed SMPL model used only during generation.
        gender: Configured SMPL gender used only during generation.
        source_model: Existing SMPL instance, primarily for repeated builds.
        smplx_model: Neutral SMPL-X reference defining the canonical rest skeleton.
    """
    if source_model is not None and (model_path is not None or gender is not None):
        raise ValueError("Pass source_model or model_path/gender, not both.")
    smpl = source_model or SMPL(model_path=model_path, gender=gender)
    smplx = smplx_model or SMPLX(gender="neutral", flat_hand_mean=True)
    shape = np.asarray(shape, dtype=np.float32)
    if shape.ndim != 1 or not 1 <= shape.shape[0] <= 10:
        raise ValueError(f"shape must have shape [1..10], got {shape.shape}.")

    target_identity = smpl.prepare_identity(shape)
    neutral_identity = smpl.prepare_identity(np.zeros_like(shape))
    raw_neutral_offsets = np.asarray(neutral_identity["local_joint_offsets"])
    measured_offsets = np.asarray(target_identity["local_joint_offsets"])
    authored_offsets = _length_only_offsets(raw_neutral_offsets, raw_neutral_offsets)
    neutral_root = np.asarray(neutral_identity["rest_joints"])[0].copy()
    neutral_root[0] = 0.0
    authored_joints = _joints_from_offsets(neutral_root, authored_offsets)

    source_path = Path(str(files("body_models.smpl_humanoid") / _SOURCE_ASSET))
    source_geometries = _load_source_geometries(source_path)
    authored_joints, _, _ = _add_finger_skeleton(
        source_geometries,
        authored_positions=_load_authored_joint_positions(source_path),
        neutral_joints=authored_joints,
        target_joints=authored_joints,
        local_offsets=authored_offsets,
    )
    neutral_offsets = _smplx_reference_offsets(smplx)
    local_offsets = _shape_scaled_offsets(
        neutral_offsets,
        smpl_neutral_offsets=raw_neutral_offsets,
        smpl_measured_offsets=measured_offsets,
    )
    neutral_joints = _joints_from_offsets(np.zeros(3), neutral_offsets, parents=ROBOT_PARENTS)
    target_joints = _joints_from_offsets(np.zeros(3), local_offsets, parents=ROBOT_PARENTS)
    parts = _rigidify_character(
        source_geometries,
        authored_joints=authored_joints,
        neutral_joints=neutral_joints,
        target_joints=target_joints,
    )

    output_path = Path(output_path)
    asset_dir = output_path.parent / f"{output_path.stem}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for old_asset in asset_dir.iterdir():
        if old_asset.is_file() and old_asset.suffix.lower() in {".obj", ".stl"}:
            old_asset.unlink()
    root = _build_xml(
        parts=parts,
        local_offsets=local_offsets,
        shape=shape,
        asset_dir=asset_dir,
        xml_dir=output_path.parent,
    )
    ET.indent(root, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="unicode", xml_declaration=False)
    return output_path


def _load_source_geometries(path: Path) -> list[_SourceGeometry]:
    scene = trimesh.load_scene(path)
    source = []
    for raw_node_name in scene.graph.nodes_geometry:
        node_name = str(raw_node_name)
        transform, geometry_name = scene.graph[node_name]
        geometry_name = str(geometry_name)
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        authored = _AUTHORED_NAME.match(geometry_name)
        if authored is None:
            raise ValueError(f"Rigid character mesh names must match J##_name__(armor|joint): {geometry_name}")
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals(multibody=True)
        material = authored.group("material")
        joint_index = int(authored.group("joint"))
        if not 0 <= joint_index < len(ROBOT_JOINT_NAMES):
            raise ValueError(f"Invalid joint prefix in authored geometry: {geometry_name}")
        source.append(
            _SourceGeometry(
                name=geometry_name,
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.faces),
                color=_ARMOR if material == "armor" else _JOINT,
                collision=material == "armor",
                joint_index=joint_index,
            )
        )
    if not source:
        raise ValueError(f"Authored character contains no mesh geometry: {path}")
    return source


def _load_authored_joint_positions(path: Path) -> dict[int, np.ndarray]:
    with path.open("rb") as file:
        file.seek(12)
        json_length, _ = struct.unpack("<II", file.read(8))
        document = json.loads(file.read(json_length))
    positions = {}
    for node in document["nodes"]:
        extras = node.get("extras", {})
        if "smpl_joint_position" in extras:
            position = np.asarray(extras["smpl_joint_position"])
            positions[int(extras["smpl_joint_index"])] = _BLENDER_TO_MODEL @ position
    return positions


def _rigidify_character(
    source: list[_SourceGeometry],
    *,
    authored_joints: np.ndarray,
    neutral_joints: np.ndarray,
    target_joints: np.ndarray,
) -> list[_RigidPart]:
    """Retarget explicitly owned rigid meshes to the SMPL-X skeleton."""
    shoulder_centers = {
        suffix: 0.5 * (ball.vertices.min(axis=0) + ball.vertices.max(axis=0))
        for suffix in ("L", "R")
        for ball in source
        if f"shoulder_ball_{suffix}" in ball.name
    }
    neutral_transforms = [
        _joint_transform(
            joint_index,
            source_joints=authored_joints,
            target_joints=neutral_joints,
        )
        for joint_index in range(len(ROBOT_JOINT_NAMES))
    ]
    shape_transforms = [
        _length_only_transform(
            joint_index,
            neutral_joints=neutral_joints,
            target_joints=target_joints,
            parents=ROBOT_PARENTS,
        )
        for joint_index in range(len(ROBOT_JOINT_NAMES))
    ]
    parts = []
    for geometry_index, geometry in enumerate(source):
        joint_index = geometry.joint_index
        joint_name = ROBOT_JOINT_NAMES[joint_index]
        authored_anchor = authored_joints[joint_index]
        neutral_anchor = neutral_joints[joint_index]
        neutral_transform = neutral_transforms[joint_index]
        anchor_name = _PRESERVED_MESH_FRAMES.get(joint_name)
        if anchor_name is not None:
            anchor_index = ROBOT_JOINT_NAMES.index(anchor_name)
            authored_anchor = authored_joints[anchor_index]
            neutral_anchor = neutral_joints[anchor_index]
            neutral_transform = np.eye(3)
        local_vertices = geometry.vertices - authored_anchor
        local_vertices += neutral_anchor - neutral_joints[joint_index]
        if "_ball_" in geometry.name:
            left, _, right = np.linalg.svd(neutral_transform)
            transform = left @ right
        else:
            transform = shape_transforms[joint_index] @ neutral_transform
        local_vertices = local_vertices @ transform.T
        socket_weights = None
        if "upper_arm_L" in geometry.name or "upper_arm_R" in geometry.name:
            suffix = "L" if "upper_arm_L" in geometry.name else "R"
            socket_distance = np.linalg.norm(geometry.vertices - shoulder_centers[suffix], axis=1)
            socket_weights = np.clip((0.050 - socket_distance) / 0.002, 0.0, 1.0)
            socket_weights = socket_weights**2 * (3.0 - 2.0 * socket_weights)
        parts.append(
            _RigidPart(
                name=f"{geometry.name}_{geometry_index}",
                joint_index=joint_index,
                vertices=local_vertices,
                faces=geometry.faces,
                color=geometry.color,
                collision=geometry.collision,
                shoulder_socket_weights=socket_weights,
            )
        )
    _align_upper_arms_to_shoulders(parts, target_joints)
    _conform_shoulder_sockets(parts, target_joints)
    _center_spine_bearings(parts, target_joints)
    _align_palms_to_fingers(parts, target_joints)
    _connect_thumb_roots(parts, target_joints)
    return parts


def _align_upper_arms_to_shoulders(parts: list[_RigidPart], joints: np.ndarray) -> None:
    """Blend each proximal upper-arm centerline onto its shoulder ball."""
    for suffix in ("L", "R"):
        ball = _find_part(parts, f"shoulder_ball_{suffix}")
        arm = _find_part(parts, f"upper_arm_{suffix}")
        ball_vertices = ball.vertices + joints[ball.joint_index]
        center = 0.5 * (ball_vertices.min(axis=0) + ball_vertices.max(axis=0))

        vertices = arm.vertices + joints[arm.joint_index]
        direction = np.sign(vertices[:, 0].mean() - center[0])
        longitudinal = direction * vertices[:, 0]
        weight = (longitudinal - longitudinal.min()) / np.ptp(longitudinal)
        proximal = weight < 0.08
        offset = center[1:] - vertices[proximal, 1:].mean(axis=0)
        blend = np.clip(1.0 - weight / 0.45, 0.0, 1.0)
        blend = blend**2 * (3.0 - 2.0 * blend)
        vertices[:, 1:] += blend[:, None] * offset
        arm.vertices[:] = vertices - joints[arm.joint_index]


def _conform_shoulder_sockets(parts: list[_RigidPart], joints: np.ndarray) -> None:
    """Seat the torso and upper arms around the final spherical shoulder balls."""
    chest = _find_part(parts, "chest_shell")
    for suffix in ("L", "R"):
        ball = _find_part(parts, f"shoulder_ball_{suffix}")
        arm = _find_part(parts, f"upper_arm_{suffix}")
        ball_vertices = ball.vertices + joints[ball.joint_index]
        center = 0.5 * (ball_vertices.min(axis=0) + ball_vertices.max(axis=0))
        ball_radius = np.linalg.norm(ball_vertices - center, axis=1).max()
        for part, clearance in ((chest, 0.005), (arm, 0.0015)):
            socket_radius = ball_radius + clearance
            world_vertices = part.vertices + joints[part.joint_index]
            delta = world_vertices - center
            if part is chest:
                side = np.sign(center[0])
                transverse = np.linalg.norm(delta[:, 1:], axis=1)
                blend = np.clip((0.110 - transverse) / 0.055, 0.0, 1.0)
                blend = blend**2 * (3.0 - 2.0 * blend)
                shoulder_edge = abs(center[0]) - 0.039
                curvature = np.clip(transverse / 0.075, 0.0, 1.0)
                curved_edge = shoulder_edge + 0.008 * curvature**1.7
                excess = np.maximum(side * world_vertices[:, 0] - curved_edge, 0.0)
                world_vertices[:, 0] -= side * blend * excess
                delta = world_vertices - center
            distance = np.linalg.norm(delta, axis=1)
            if part.shoulder_socket_weights is not None:
                scale = socket_radius / np.maximum(distance, 1e-9)
                socket_vertices = center + delta * scale[:, None]
                weight = part.shoulder_socket_weights[:, None]
                world_vertices = world_vertices * (1.0 - weight) + socket_vertices * weight
                delta = world_vertices - center
                distance = np.linalg.norm(delta, axis=1)
            intruding = distance < socket_radius
            scale = socket_radius / np.maximum(distance[intruding], 1e-9)
            world_vertices[intruding] = center + delta[intruding] * scale[:, None]
            part.vertices[:] = world_vertices - joints[part.joint_index]


def _center_spine_bearings(parts: list[_RigidPart], joints: np.ndarray) -> None:
    """Align and center each bearing between its neighboring body shells."""
    vertical_axis = 1
    configurations = (
        ("lower_spine_bearing", "pelvis_shell", "abdomen_shell"),
        ("upper_spine_bearing", "abdomen_shell", "chest_shell"),
    )
    for bearing_name, lower_name, upper_name in configurations:
        bearing = _find_part(parts, bearing_name)
        lower = _find_part(parts, lower_name)
        upper = _find_part(parts, upper_name)
        bearing_world = bearing.vertices + joints[bearing.joint_index]
        lower_world = lower.vertices + joints[lower.joint_index]
        upper_world = upper.vertices + joints[upper.joint_index]

        lower_edge = lower_world[:, vertical_axis].max()
        upper_edge = upper_world[:, vertical_axis].min()
        lower_normal = _interface_normal(lower_world, upper=True)
        upper_normal = _interface_normal(upper_world, upper=False)
        target_normal = lower_normal + upper_normal
        target_normal /= np.linalg.norm(target_normal)

        bearing_center = bearing_world.mean(axis=0)
        bearing_offsets = bearing_world - bearing_center
        _, bearing_axes = np.linalg.eigh(bearing_offsets.T @ bearing_offsets)
        bearing_normal = bearing_axes[:, 0]
        bearing_normal *= np.sign(bearing_normal[vertical_axis])
        rotation, _ = scipy_transform.Rotation.align_vectors(
            target_normal[None],
            bearing_normal[None],
        )
        bearing_offsets = bearing_offsets @ rotation.as_matrix().T

        normal_coordinate = bearing_offsets @ target_normal
        tangent = bearing_offsets - normal_coordinate[:, None] * target_normal
        desired_height = max(upper_edge - lower_edge - 0.006, 0.004)
        if np.ptp(bearing_offsets[:, vertical_axis]) > desired_height:
            low, high = 0.0, 1.0
            for _ in range(32):
                scale = 0.5 * (low + high)
                vertical = tangent[:, vertical_axis] + scale * normal_coordinate * target_normal[vertical_axis]
                if np.ptp(vertical) > desired_height:
                    high = scale
                else:
                    low = scale
            bearing_offsets = tangent + low * normal_coordinate[:, None] * target_normal
        bearing_world = bearing_center + bearing_offsets

        midpoint = 0.5 * (lower_edge + upper_edge)
        center = 0.5 * (bearing_world[:, vertical_axis].min() + bearing_world[:, vertical_axis].max())
        bearing_world[:, vertical_axis] += midpoint - center
        bearing.vertices[:] = bearing_world - joints[bearing.joint_index]


def _interface_normal(vertices: np.ndarray, *, upper: bool) -> np.ndarray:
    """Estimate a torso shell interface normal from its boundary vertices."""
    vertical = vertices[:, 1]
    edge = vertical.max() if upper else vertical.min()
    selected = vertices[np.abs(vertical - edge) < 0.004]
    centered = selected - selected.mean(axis=0)
    _, axes = np.linalg.eigh(centered.T @ centered)
    normal = axes[:, 0]
    normal *= np.sign(normal[1])
    return normal


def _find_part(parts: list[_RigidPart], name: str) -> _RigidPart:
    return next(part for part in parts if name in part.name)


def _align_palms_to_fingers(parts: list[_RigidPart], joints: np.ndarray) -> None:
    """Taper each distal palm onto its first row of finger bearings."""
    for suffix in ("L", "R"):
        palm = _find_part(parts, f"palm_{suffix}")
        bearings = [part for part in parts if f"_1_bearing_{suffix}" in part.name and "thumb" not in part.name]
        bearing_centers = [(part.vertices + joints[part.joint_index]).mean(axis=0) for part in bearings]
        target_height = np.mean(bearing_centers, axis=0)[1]

        vertices = palm.vertices + joints[palm.joint_index]
        palm_center = vertices.mean(axis=0)
        finger_center = np.mean(bearing_centers, axis=0)
        direction = np.sign(finger_center[0] - palm_center[0])
        longitudinal = direction * vertices[:, 0]
        weight = (longitudinal - longitudinal.min()) / np.ptp(longitudinal)
        tip = weight > 0.92
        tip_height = vertices[tip, 1].mean()

        blend = np.clip((weight - 0.35) / 0.65, 0.0, 1.0)
        blend = blend**2 * (3.0 - 2.0 * blend)
        shift = target_height - tip_height
        vertices[:, 1] += shift * blend

        taper = np.clip((weight - 0.65) / 0.35, 0.0, 1.0)
        taper = taper**2 * (3.0 - 2.0 * taper)
        centerline = tip_height + shift * blend
        vertices[:, 1] = centerline + (vertices[:, 1] - centerline) * (1.0 - 0.25 * taper)
        palm.vertices[:] = vertices - joints[palm.joint_index]


def _connect_thumb_roots(parts: list[_RigidPart], joints: np.ndarray) -> None:
    """Grow a localized thenar lobe from each palm to its thumb root."""
    for suffix in ("L", "R"):
        palm = _find_part(parts, f"palm_{suffix}")
        thumb = _find_part(parts, f"thumb_proximal_{suffix}")
        palm_vertices = palm.vertices + joints[palm.joint_index]
        thumb_vertices = thumb.vertices + joints[thumb.joint_index]

        distances = np.linalg.norm(
            palm_vertices[:, None, :] - thumb_vertices[None, :, :],
            axis=2,
        )
        palm_index, thumb_index = np.unravel_index(np.argmin(distances), distances.shape)
        origin = palm_vertices[palm_index]
        extension = thumb_vertices[thumb_index] - origin
        gap = np.linalg.norm(extension)
        if gap <= 0.001:
            continue
        extension *= (gap - 0.001) / gap

        radius = 0.022
        distance_from_origin = np.linalg.norm(palm_vertices - origin, axis=1)
        weight = np.clip(1.0 - distance_from_origin / radius, 0.0, 1.0)
        weight = weight**2 * (3.0 - 2.0 * weight)
        palm_vertices += weight[:, None] * extension
        palm.vertices[:] = palm_vertices - joints[palm.joint_index]


def _smplx_reference_offsets(smplx: SMPLX) -> np.ndarray:
    identity = smplx.prepare_identity(
        np.zeros(10, dtype=np.float32),
        expression=np.zeros(10, dtype=np.float32),
    )
    smplx_joints = np.asarray(identity["rest_joints"])
    joints = np.zeros((len(ROBOT_JOINT_NAMES), 3), dtype=smplx_joints.dtype)
    joints[:22] = smplx_joints[:22]
    joints[22] = smplx_joints[20]
    joints[23] = smplx_joints[21]
    for joint_index, name in enumerate(ROBOT_JOINT_NAMES[24:], start=24):
        joints[joint_index] = smplx_joints[smplx.joint_names.index(name)]
    joints -= joints[0]

    offsets = np.zeros_like(joints)
    for joint_index in range(1, len(joints)):
        offsets[joint_index] = joints[joint_index] - joints[ROBOT_PARENTS[joint_index]]
    return _length_only_offsets(offsets, offsets)


def _shape_scaled_offsets(
    reference_offsets: np.ndarray,
    *,
    smpl_neutral_offsets: np.ndarray,
    smpl_measured_offsets: np.ndarray,
) -> np.ndarray:
    desired_offsets = reference_offsets.copy()
    for joint_index in range(1, len(JOINT_NAMES) - 2):
        neutral_length = np.linalg.norm(smpl_neutral_offsets[joint_index])
        measured_length = np.linalg.norm(smpl_measured_offsets[joint_index])
        desired_offsets[joint_index] *= measured_length / neutral_length
    return _length_only_offsets(reference_offsets, desired_offsets)


def _joint_transform(
    joint_index: int,
    *,
    source_joints: np.ndarray,
    target_joints: np.ndarray,
) -> np.ndarray:
    children = [index for index, parent in enumerate(ROBOT_PARENTS) if parent == joint_index]
    if children:
        centerline_children = [
            index
            for index in children
            if abs(source_joints[index, 0] - source_joints[joint_index, 0]) < 1e-8
            and abs(target_joints[index, 0] - target_joints[joint_index, 0]) < 1e-8
        ]
        related_joint = max(
            centerline_children or children,
            key=lambda index: float(np.linalg.norm(source_joints[index] - source_joints[joint_index])),
        )
        source_axis = source_joints[related_joint] - source_joints[joint_index]
        target_axis = target_joints[related_joint] - target_joints[joint_index]
        scale_to_child = True
    else:
        parent = ROBOT_PARENTS[joint_index]
        if parent < 0:
            return np.eye(3)
        source_axis = source_joints[joint_index] - source_joints[parent]
        target_axis = target_joints[joint_index] - target_joints[parent]
        scale_to_child = False

    source_length = np.linalg.norm(source_axis)
    target_length = np.linalg.norm(target_axis)
    if source_length <= 1e-8 or target_length <= 1e-8:
        return np.eye(3)
    longitudinal_scale = target_length / source_length if scale_to_child else 1.0
    source_direction = source_axis / source_length
    target_direction = target_axis / target_length
    rotation, _ = scipy_transform.Rotation.align_vectors(
        target_direction[None],
        source_direction[None],
    )
    scale = np.eye(3) + (longitudinal_scale - 1.0) * np.outer(
        source_direction,
        source_direction,
    )
    return rotation.as_matrix() @ scale


def _length_only_offsets(neutral_offsets: np.ndarray, measured_offsets: np.ndarray) -> np.ndarray:
    """Apply measured lengths to an exactly bilateral neutral skeleton."""
    result = neutral_offsets.copy()
    bilateral_pairs = [pair for pair in _BILATERAL_JOINT_PAIRS if pair[1] < len(result)]
    paired_joints = {joint for pair in bilateral_pairs for joint in pair}
    for left_index, right_index in bilateral_pairs:
        reflected_right = neutral_offsets[right_index] * _SAGITTAL_REFLECTION
        average_direction = neutral_offsets[left_index] + reflected_right
        direction_length = float(np.linalg.norm(average_direction))
        if direction_length <= 1e-8:
            average_direction = neutral_offsets[left_index]
            direction_length = float(np.linalg.norm(average_direction))
        if direction_length <= 1e-8:
            result[left_index] = 0.0
            result[right_index] = 0.0
            continue
        average_length = 0.5 * (
            float(np.linalg.norm(measured_offsets[left_index])) + float(np.linalg.norm(measured_offsets[right_index]))
        )
        left_offset = average_direction * average_length / direction_length
        result[left_index] = left_offset
        result[right_index] = left_offset * _SAGITTAL_REFLECTION

    for joint_index in range(1, len(result)):
        if joint_index in paired_joints:
            continue
        result[joint_index, 0] = 0.0
        neutral_length = float(np.linalg.norm(result[joint_index]))
        measured_length = float(np.linalg.norm(measured_offsets[joint_index]))
        if neutral_length > 1e-8:
            result[joint_index] *= measured_length / neutral_length
    result[0] = 0.0
    return result


def _joints_from_offsets(
    root_joint: np.ndarray,
    local_offsets: np.ndarray,
    *,
    parents: list[int] = PARENTS,
) -> np.ndarray:
    joints = np.empty_like(local_offsets)
    joints[0] = root_joint
    for joint_index in range(1, len(joints)):
        joints[joint_index] = joints[parents[joint_index]] + local_offsets[joint_index]
    return joints


def _add_finger_skeleton(
    source: list[_SourceGeometry],
    *,
    authored_positions: dict[int, np.ndarray],
    neutral_joints: np.ndarray,
    target_joints: np.ndarray,
    local_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append the fixed-size SMPL-X finger chains from their authored bearings."""
    authored_centers = dict(authored_positions)
    for joint_index in range(len(JOINT_NAMES), len(ROBOT_JOINT_NAMES)):
        if joint_index in authored_centers:
            continue
        bearings = [
            geometry for geometry in source if geometry.joint_index == joint_index and "_bearing_" in geometry.name
        ]
        if len(bearings) != 1:
            name = ROBOT_JOINT_NAMES[joint_index]
            raise ValueError(f"Expected one authored bearing for finger joint {name}, found {len(bearings)}.")
        bearing = bearings[0]
        authored_centers[joint_index] = 0.5 * (bearing.vertices.min(axis=0) + bearing.vertices.max(axis=0))

    robot_joints = np.zeros((len(ROBOT_JOINT_NAMES), 3), dtype=neutral_joints.dtype)
    target_robot_joints = np.zeros_like(robot_joints)
    robot_offsets = np.zeros_like(robot_joints)
    robot_joints[: len(JOINT_NAMES)] = neutral_joints
    target_robot_joints[: len(JOINT_NAMES)] = target_joints
    robot_offsets[: len(JOINT_NAMES)] = local_offsets
    for joint_index in range(len(JOINT_NAMES), len(ROBOT_JOINT_NAMES)):
        parent = ROBOT_PARENTS[joint_index]
        robot_joints[joint_index] = authored_centers[joint_index]
        fixed_offset = authored_centers[joint_index] - robot_joints[parent]
        target_robot_joints[joint_index] = target_robot_joints[parent] + fixed_offset
        robot_offsets[joint_index] = fixed_offset
    return robot_joints, target_robot_joints, robot_offsets


def _length_only_transform(
    joint_index: int,
    *,
    neutral_joints: np.ndarray,
    target_joints: np.ndarray,
    parents: list[int] = PARENTS,
) -> np.ndarray:
    children = [index for index, parent in enumerate(parents) if parent == joint_index]
    if not children:
        return np.eye(3)
    child = max(children, key=lambda index: float(np.linalg.norm(neutral_joints[index] - neutral_joints[joint_index])))
    neutral_axis = neutral_joints[child] - neutral_joints[joint_index]
    target_axis = target_joints[child] - target_joints[joint_index]
    neutral_length = float(np.linalg.norm(neutral_axis))
    target_length = float(np.linalg.norm(target_axis))
    if neutral_length <= 1e-8:
        return np.eye(3)
    direction = neutral_axis / neutral_length
    return np.eye(3) + (target_length / neutral_length - 1.0) * np.outer(direction, direction)


def _build_xml(
    *,
    parts: list[_RigidPart],
    local_offsets: np.ndarray,
    shape: np.ndarray,
    asset_dir: Path,
    xml_dir: Path,
) -> ET.Element:
    root = ET.Element("mujoco", model="smpl_robot")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep="0.008333", gravity="0 -9.81 0", integrator="implicitfast")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="2", armature="0.02", limited="true")
    ET.SubElement(default, "geom", friction="0.8 0.1 0.1")
    asset = ET.SubElement(root, "asset")
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "numeric", name="smpl_shape", data=_vec(shape))

    parts_by_joint: dict[int, list[tuple[_RigidPart, str]]] = {index: [] for index in range(len(ROBOT_JOINT_NAMES))}
    for part_index, part in enumerate(parts):
        mesh_name = f"{ROBOT_JOINT_NAMES[part.joint_index]}_{part.name}_{part_index}"
        mesh_path = asset_dir / f"{mesh_name}.obj"
        part.mesh.export(mesh_path)
        ET.SubElement(asset, "mesh", name=mesh_name, file=mesh_path.relative_to(xml_dir).as_posix())
        parts_by_joint[part.joint_index].append((part, mesh_name))

    children = {index: [] for index in range(len(ROBOT_JOINT_NAMES))}
    for index, parent in enumerate(ROBOT_PARENTS):
        if parent >= 0:
            children[parent].append(index)
    worldbody = ET.SubElement(root, "worldbody")

    def add_body(joint_index: int, parent_element: ET.Element) -> None:
        name = ROBOT_JOINT_NAMES[joint_index]
        body = ET.SubElement(parent_element, "body", name=name, pos=_vec(local_offsets[joint_index]))
        if not parts_by_joint[joint_index] and (joint_index == 0 or name in _BODY_JOINTS):
            ET.SubElement(
                body,
                "inertial",
                pos="0 0 0",
                mass="0.001",
                diaginertia="0.000001 0.000001 0.000001",
            )
        if joint_index == 0:
            ET.SubElement(body, "freejoint", name="root")
        elif name in _ACTUATED_JOINTS:
            for axis_index, axis in enumerate(("x", "y", "z")):
                vector = ["0", "0", "0"]
                vector[axis_index] = "1"
                ET.SubElement(
                    body,
                    "joint",
                    name=f"{name}_{axis}",
                    type="hinge",
                    axis=" ".join(vector),
                    range="-3.14159 3.14159",
                )
        for part, mesh_name in parts_by_joint[joint_index]:
            collision = "1" if part.collision else "0"
            ET.SubElement(
                body,
                "geom",
                name=mesh_name,
                type="mesh",
                mesh=mesh_name,
                rgba=part.color,
                contype=collision,
                conaffinity=collision,
            )
        for child in children[joint_index]:
            add_body(child, body)

    add_body(0, worldbody)
    return root


def _vec(value: np.ndarray) -> str:
    return " ".join(f"{float(component):.8g}" for component in np.ravel(value))


__all__ = ["generate_smpl_robot"]
