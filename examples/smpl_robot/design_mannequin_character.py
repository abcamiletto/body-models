"""Author the smooth, non-skinned SMPL mannequin in Blender.

The output contains independent rigid meshes with direct SMPL joint ownership.
There is no armature, skin, animation, modifier, or shape key in the deliverable.
"""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "src/body_models/robots/smpl_humanoid/assets"
ARTIFACT_DIR = ROOT / "artifacts/smpl_robot"
BLEND_PATH = ASSET_DIR / "smpl_robot_professional.blend"
GLB_PATH = ASSET_DIR / "smpl_robot_professional.glb"

# Near-monochrome, editable mannequin palette. The joints are deliberately only
# a little darker so the model reads as one object, like the supplied reference.
ARMOR_COLOR = (0.68, 0.48, 0.24, 1.0)
JOINT_COLOR = (0.52, 0.34, 0.16, 1.0)
# ManoSim's neutral MANO hand spans 174.7 mm from wrist to middle fingertip.
# Preserve the sculpt's width and thickness, but correct its overlong silhouette.
HAND_LENGTH_SCALE = 0.8135
SOCKET_CLEARANCE = 0.0015
SOCKET_EXPOSURE = 0.62
SHOULDER_RADIUS = 0.045
SHOULDER_SOCKET_RADIUS = 0.046
ELBOW_RADIUS = 0.035
WRIST_RADIUS = 0.026
HIP_RADIUS = 0.050
KNEE_RADIUS = 0.040
ANKLE_RADIUS = 0.031
FINGER_CHAINS = tuple(
    tuple(f"{side}_{digit}{level}" for level in (1, 2, 3))
    for side in ("L", "R")
    for digit in ("Index", "Middle", "Pinky", "Ring", "Thumb")
)
JOINT_INDEX = {
    name: index
    for index, name in enumerate(
        (name for chain in FINGER_CHAINS for name in chain),
        start=24,
    )
}


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _clear_scene()
    armor = _material(
        "Armor — edit ARMOR_COLOR",
        ARMOR_COLOR,
        metallic=0.0,
        roughness=0.30,
    )
    joint = _material(
        "Joints — edit JOINT_COLOR",
        JOINT_COLOR,
        metallic=0.0,
        roughness=0.38,
    )
    studio = _material("Studio", (0.34, 0.32, 0.29, 1.0), metallic=0.0, roughness=0.78)

    _build_character(armor, joint)
    _studio(studio)
    _balance_studio_lighting()
    world_shader = bpy.context.scene.world.node_tree.nodes["Background"]
    world_shader.inputs["Color"].default_value = (0.13, 0.105, 0.078, 1.0)
    world_shader.inputs["Strength"].default_value = 0.48
    camera = _add_camera()
    _configure_render()
    scene = bpy.context.scene
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1280

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    _export_character(GLB_PATH)
    _render(camera, ARTIFACT_DIR / "professional_beauty.png", (2.65, -4.25, 1.00), 78)
    _render(camera, ARTIFACT_DIR / "professional_front.png", (0.0, -5.2, -0.28), 66)
    _render(camera, ARTIFACT_DIR / "professional_back.png", (0.0, 5.2, -0.28), 66)
    _render(
        camera,
        ARTIFACT_DIR / "professional_joints_upper.png",
        (1.35, -3.0, 0.38),
        55,
        target=(0.0, 0.025, 0.20),
    )
    _render(
        camera,
        ARTIFACT_DIR / "professional_shoulders_closeup.png",
        (0.0, -1.45, 0.30),
        72,
        target=(0.0, 0.025, 0.22),
    )
    _render(
        camera,
        ARTIFACT_DIR / "professional_joints_lower.png",
        (0.85, -3.0, -0.70),
        72,
        target=(0.0, 0.0, -0.70),
    )
    _render(
        camera,
        ARTIFACT_DIR / "professional_hand.png",
        (1.15, -0.65, 0.34),
        75,
        target=(0.79, 0.045, 0.22),
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    print(f"Wrote rigid mannequin with {_character_mesh_count()} character meshes")


def _build_character(armor: bpy.types.Material, joint: bpy.types.Material) -> None:
    # Pelvis, abdomen, chest: three calm solids separated by narrow articulation
    # gaps. Superellipse sections make the torso broad and planar without
    # introducing hard box corners.
    pelvis = _super_loft_z(
        "J00__pelvis_shell__armor",
        0,
        [
            (-0.330, 0.044, 0.050),
            (-0.300, 0.074, 0.070),
            (-0.246, 0.122, 0.088),
            (-0.180, 0.141, 0.094),
            (-0.118, 0.118, 0.077),
        ],
        armor,
        exponent=0.78,
        bevel=0.007,
    )
    _carve_sockets(
        pelvis,
        tuple(((side * 0.082, -0.002, -0.320), HIP_RADIUS) for side in (-1.0, 1.0)),
        symmetrize=True,
    )
    _super_loft_z(
        "J03__abdomen_shell__armor",
        3,
        [
            (-0.098, 0.082, 0.060),
            (-0.076, 0.091, 0.065),
            (-0.016, 0.094, 0.068),
            (0.018, 0.088, 0.063),
        ],
        armor,
        exponent=0.80,
        bevel=0.005,
    )
    _rounded_disk(
        "J03__lower_spine_bearing__joint",
        3,
        (0.0, 0.0, -0.108),
        (0.047, 0.034),
        0.014,
        joint,
    )
    _rounded_disk(
        "J06__upper_spine_bearing__joint",
        6,
        (0.0, 0.0, 0.027),
        (0.056, 0.040),
        0.012,
        joint,
    )
    chest = _super_loft_z(
        "J09__chest_shell__armor",
        9,
        [
            (0.036, 0.112, 0.075),
            (0.055, 0.125, 0.082),
            (0.135, 0.143, 0.089),
            (0.165, 0.149, 0.091),
            (0.190, 0.152, 0.092),
            (0.208, 0.148, 0.092),
            (0.225, 0.135, 0.090),
            (0.242, 0.129, 0.087),
            (0.265, 0.124, 0.082),
            (0.282, 0.114, 0.074),
        ],
        armor,
        exponent=0.78,
        bevel=0.007,
    )
    _carve_sockets(
        chest,
        tuple(((side * 0.174, 0.016, 0.226), SHOULDER_SOCKET_RADIUS) for side in (-1.0, 1.0)),
        rim_width=0.0035,
        symmetrize=True,
    )

    # Featureless sculpted head and a visible neck pedestal.
    _super_loft_z(
        "J12__neck_pedestal__joint",
        12,
        [
            (0.290, 0.047, 0.044),
            (0.315, 0.052, 0.047),
            (0.350, 0.047, 0.043),
        ],
        joint,
        exponent=0.80,
        bevel=0.004,
    )
    _sculpted_head(
        "J15__featureless_mannequin_head__armor",
        15,
        armor,
    )

    for side, suffix, shoulder, upper, elbow, wrist, hand in (
        (1.0, "L", 13, 16, 18, 20, 22),
        (-1.0, "R", 14, 17, 19, 21, 23),
    ):
        shoulder_center = (side * 0.174, 0.016, 0.226)
        elbow_center = (side * 0.431, 0.042, 0.216)
        wrist_center = (side * 0.682, 0.045, 0.220)
        _ellipsoid(
            f"J{shoulder:02d}__shoulder_ball_{suffix}__joint",
            shoulder,
            shoulder_center,
            (SHOULDER_RADIUS,) * 3,
            joint,
        )
        upper_arm = _socketed_super_loft_x(
            f"J{upper:02d}__upper_arm_{suffix}__armor",
            upper,
            side,
            [
                (
                    0.174 + 0.24 * SHOULDER_RADIUS,
                    0.050,
                    0.052,
                ),
                (0.192, 0.052, 0.054),
                (0.202, 0.054, 0.056),
                (0.216, 0.055, 0.057),
                (0.235, 0.0545, 0.0565),
                (0.260, 0.052, 0.054),
                (0.300, 0.049, 0.051),
                (0.345, 0.046, 0.048),
                (0.390, 0.042, 0.044),
                (
                    0.431 - SOCKET_EXPOSURE * ELBOW_RADIUS,
                    0.038,
                    0.041,
                ),
            ],
            armor,
            socket_center=shoulder_center,
            socket_radius=SHOULDER_SOCKET_RADIUS + SOCKET_CLEARANCE,
            y_centers=(0.016, 0.0165, 0.017, 0.018, 0.020, 0.023, 0.028, 0.034, 0.039, 0.042),
            z_centers=(0.226, 0.226, 0.2258, 0.2255, 0.225, 0.224, 0.222, 0.220, 0.218, 0.216),
        )
        _carve_sockets(
            upper_arm,
            ((elbow_center, ELBOW_RADIUS),),
            rim_width=0.003,
        )
        _ellipsoid(
            f"J{elbow:02d}__elbow_ball_{suffix}__joint",
            elbow,
            elbow_center,
            (ELBOW_RADIUS,) * 3,
            joint,
        )
        forearm = _super_loft_x(
            f"J{elbow:02d}__forearm_{suffix}__armor",
            elbow,
            side,
            [
                (
                    0.431 + SOCKET_EXPOSURE * ELBOW_RADIUS,
                    0.035,
                    0.038,
                ),
                (0.500, 0.043, 0.045),
                (0.575, 0.041, 0.043),
                (
                    0.682 - SOCKET_EXPOSURE * WRIST_RADIUS,
                    0.031,
                    0.033,
                ),
            ],
            armor,
            z_center=0.220,
            y_centers=(0.042, 0.043, 0.044, 0.045),
        )
        _carve_sockets(
            forearm,
            (
                (elbow_center, ELBOW_RADIUS),
                (wrist_center, WRIST_RADIUS),
            ),
        )
        _ellipsoid(
            f"J{wrist:02d}__wrist_ball_{suffix}__joint",
            wrist,
            wrist_center,
            (WRIST_RADIUS,) * 3,
            joint,
        )
        _build_hand(side, suffix, hand, armor, joint, wrist_center)

    for side, suffix, hip, knee, ankle, toe in (
        (1.0, "L", 1, 4, 7, 10),
        (-1.0, "R", 2, 5, 8, 11),
    ):
        hip_center = (side * 0.082, -0.002, -0.320)
        knee_center = (side * 0.103, 0.0, -0.691)
        ankle_center = (side * 0.090, 0.0, -1.084)
        _ellipsoid(
            f"J{hip:02d}__hip_ball_{suffix}__joint",
            hip,
            hip_center,
            (HIP_RADIUS,) * 3,
            joint,
        )
        thigh = _super_loft_z_offset(
            f"J{hip:02d}__thigh_{suffix}__armor",
            hip,
            side,
            [
                (
                    -0.320 - SOCKET_EXPOSURE * HIP_RADIUS,
                    0.057,
                    0.057,
                    0.083,
                ),
                (-0.390, 0.066, 0.064, 0.087),
                (-0.500, 0.065, 0.063, 0.094),
                (-0.600, 0.055, 0.055, 0.099),
                (
                    -0.691 + SOCKET_EXPOSURE * KNEE_RADIUS,
                    0.044,
                    0.047,
                    0.101,
                ),
            ],
            armor,
        )
        _carve_sockets(
            thigh,
            (
                (hip_center, HIP_RADIUS),
                (knee_center, KNEE_RADIUS),
            ),
        )
        _ellipsoid(
            f"J{knee:02d}__knee_ball_{suffix}__joint",
            knee,
            knee_center,
            (KNEE_RADIUS,) * 3,
            joint,
        )
        shin = _super_loft_z_offset(
            f"J{knee:02d}__shin_{suffix}__armor",
            knee,
            side,
            [
                (
                    -0.691 - SOCKET_EXPOSURE * KNEE_RADIUS,
                    0.041,
                    0.041,
                    0.103,
                ),
                (-0.775, 0.049, 0.048, 0.102),
                (-0.875, 0.053, 0.053, 0.099),
                (-0.990, 0.045, 0.046, 0.094),
                (
                    -1.084 + SOCKET_EXPOSURE * ANKLE_RADIUS,
                    0.035,
                    0.038,
                    0.090,
                ),
            ],
            armor,
        )
        _carve_sockets(
            shin,
            (
                (knee_center, KNEE_RADIUS),
                (ankle_center, ANKLE_RADIUS),
            ),
        )
        _ellipsoid(
            f"J{ankle:02d}__ankle_ball_{suffix}__joint",
            ankle,
            ankle_center,
            (ANKLE_RADIUS,) * 3,
            joint,
        )
        rear_foot = _super_loft_y(
            f"J{ankle:02d}__rear_foot_{suffix}__armor",
            ankle,
            1.0,
            [
                (0.058, 0.034, 0.022, -1.152),
                (0.040, 0.044, 0.0305, -1.1465),
                (0.015, 0.048, 0.0445, -1.1355),
                (-0.025, 0.050, 0.0395, -1.1405),
                (-0.065, 0.051, 0.034, -1.146),
                (-0.100, 0.050, 0.027, -1.151),
                (-0.130, 0.046, 0.019, -1.154),
            ],
            armor,
            exponent=0.68,
            x_center=0.093,
        )
        _carve_sockets(
            rear_foot,
            (((0.090, 0.0, -1.084), ANKLE_RADIUS),),
        )
        if side < 0.0:
            for vertex in rear_foot.data.vertices:
                vertex.co.x *= -1.0
            for polygon in rear_foot.data.polygons:
                polygon.flip()
        _ellipsoid(
            f"J{toe:02d}__toe_pivot_{suffix}__joint",
            toe,
            (side * 0.093, -0.140, -1.148),
            (0.0005, 0.0005, 0.0005),
            joint,
            segments=8,
            rings=4,
        )
        _super_loft_y(
            f"J{toe:02d}__forefoot_{suffix}__armor",
            toe,
            side,
            [
                (-0.136, 0.046, 0.0195, -1.1535),
                (-0.150, 0.045, 0.0195, -1.1555),
                (-0.168, 0.042, 0.0175, -1.1575),
                (-0.184, 0.037, 0.014, -1.160),
                (-0.195, 0.030, 0.0095, -1.1615),
                (-0.205, 0.022, 0.008, -1.160),
            ],
            armor,
            exponent=0.68,
            x_center=0.093,
        )


def _balance_studio_lighting() -> None:
    """Keep front-review highlights bilateral so shading cannot mimic skew."""
    target = Vector((0.0, 0.0, -0.28))
    for name, x in (("Key", -2.7), ("Fill", 2.7)):
        light = bpy.data.objects[name]
        light.data.energy = 900.0
        light.data.size = 3.1
        light.data.color = (1.0, 0.91, 0.82)
        light.location = (x, -3.0, 2.7)
        light.rotation_euler = (target - light.location).to_track_quat("-Z", "Y").to_euler()


def _build_hand(
    side: float,
    suffix: str,
    joint_index: int,
    armor: bpy.types.Material,
    joint: bpy.types.Material,
    wrist_center: tuple[float, float, float],
) -> None:
    existing_objects = set(bpy.data.objects)
    wrist_center_vector = Vector(wrist_center)
    palm = _palm_mesh(
        f"J{joint_index:02d}__palm_{suffix}__armor",
        joint_index,
        side,
        [
            (
                0.682 + SOCKET_EXPOSURE * WRIST_RADIUS,
                0.015,
                0.021,
            ),
            (0.716, 0.0148, 0.026),
            (0.738, 0.0145, 0.032),
            (0.758, 0.0142, 0.035),
            (0.778, 0.0135, 0.034),
            (0.792, 0.0115, 0.029),
            (0.802, 0.0093, 0.026),
        ],
        armor,
        y_centers=(0.052, 0.052, 0.0515, 0.051, 0.0505, 0.050, 0.0495),
        z_centers=(0.220, 0.220, 0.218, 0.216, 0.217, 0.220, 0.221),
        palmar_fullness=(0.0, 0.001, 0.0025, 0.0035, 0.003, 0.0015, 0.0),
        thumb_reach=(0.0, 0.001, 0.004, 0.007, 0.008, 0.006, 0.0025),
        distal_contour=(
            (0.193, 0.793),
            (0.203, 0.799),
            (0.209, 0.795),
            (0.215, 0.800),
            (0.221, 0.796),
            (0.227, 0.803),
            (0.233, 0.796),
            (0.239, 0.801),
            (0.249, 0.793),
        ),
    )
    straight_digits = {
        "Index": ((0.800, 0.829, 0.857, 0.882), 0.239, 0.0090),
        "Middle": ((0.800, 0.832, 0.863, 0.891), 0.227, 0.0092),
        "Ring": ((0.800, 0.830, 0.858, 0.884), 0.215, 0.0087),
        "Pinky": ((0.799, 0.823, 0.845, 0.866), 0.203, 0.0078),
    }
    segment_names = ("proximal", "middle", "distal")
    for digit, (x_positions, z, radius) in straight_digits.items():
        points = [(side * x, 0.058, z) for x in x_positions]
        joint_indices = [JOINT_INDEX[f"{suffix}_{digit}{level}"] for level in (1, 2, 3)]
        for segment_index, (finger_joint, start, end) in enumerate(
            zip(joint_indices, points[:-1], points[1:], strict=True)
        ):
            bearing_radius = radius * (0.82 - 0.08 * segment_index)
            _ellipsoid(
                f"J{finger_joint:02d}__{digit.lower()}_{segment_index + 1}_bearing_{suffix}__joint",
                finger_joint,
                start,
                (bearing_radius, bearing_radius * 0.92, bearing_radius),
                joint,
                segments=32,
                rings=16,
            )
            _rounded_tapered_segment(
                f"J{finger_joint:02d}__{digit.lower()}_{segment_names[segment_index]}_{suffix}__armor",
                finger_joint,
                _inset_segment_endpoint(start, end, 0.0025),
                _inset_segment_endpoint(end, start, 0.0025),
                radius * (1.0 - 0.12 * segment_index),
                radius * (0.88 - 0.12 * segment_index),
                armor,
            )
        fingertip_joint = joint_indices[-1]
        fingertip_radius = radius * 0.62
        _ellipsoid(
            f"J{fingertip_joint:02d}__{digit.lower()}_fingertip_{suffix}__armor",
            fingertip_joint,
            points[-1],
            (fingertip_radius, fingertip_radius * 0.92, fingertip_radius),
            armor,
            segments=32,
            rings=16,
        )

    thumb_points = [
        (side * 0.773, 0.054, 0.199),
        (side * 0.797, 0.050, 0.190),
        (side * 0.818, 0.047, 0.182),
        (side * 0.840, 0.044, 0.176),
    ]
    thumb_indices = [JOINT_INDEX[f"{suffix}_Thumb{level}"] for level in (1, 2, 3)]
    for segment_index, (finger_joint, start, end) in enumerate(
        zip(thumb_indices, thumb_points[:-1], thumb_points[1:], strict=True)
    ):
        radius = 0.0092 * (1.0 - 0.13 * segment_index)
        if segment_index:
            _ellipsoid(
                f"J{finger_joint:02d}__thumb_{segment_index + 1}_bearing_{suffix}__joint",
                finger_joint,
                start,
                (radius * 0.82, radius * 0.76, radius * 0.82),
                joint,
                segments=32,
                rings=16,
            )
        segment = _rounded_tapered_segment(
            f"J{finger_joint:02d}__thumb_{segment_names[segment_index]}_{suffix}__armor",
            finger_joint,
            _inset_segment_endpoint(start, end, 0.0022),
            _inset_segment_endpoint(end, start, 0.0022),
            radius,
            radius * 0.84,
            armor,
        )
        if segment_index == 0:
            segment["smpl_joint_position"] = start
    thumb_tip_radius = 0.0060
    _ellipsoid(
        f"J{thumb_indices[-1]:02d}__thumb_fingertip_{suffix}__armor",
        thumb_indices[-1],
        thumb_points[-1],
        (thumb_tip_radius, thumb_tip_radius * 0.90, thumb_tip_radius),
        armor,
        segments=32,
        rings=16,
    )
    # SMPL-X neutral hands are palm-down; the sculpt is authored palm-forward.
    hand_transform = (
        Matrix.Translation(wrist_center_vector)
        @ Matrix.Rotation(math.pi / 2, 4, "X")
        @ Matrix.Diagonal((HAND_LENGTH_SCALE, 1.0, 1.0, 1.0))
        @ Matrix.Translation(-wrist_center_vector)
    )
    for obj in set(bpy.data.objects) - existing_objects:
        obj.matrix_world = hand_transform @ obj.matrix_world
        if "smpl_joint_position" in obj:
            position = hand_transform @ Vector(obj["smpl_joint_position"])
            obj["smpl_joint_position"] = tuple(position)
    _carve_sockets(palm, ((wrist_center, WRIST_RADIUS),))


def _palm_mesh(
    name: str,
    joint_index: int,
    side: float,
    profile: list[tuple[float, float, float]],
    material: bpy.types.Material,
    *,
    y_centers: tuple[float, ...],
    z_centers: tuple[float, ...],
    palmar_fullness: tuple[float, ...],
    thumb_reach: tuple[float, ...],
    distal_contour: tuple[tuple[float, float], ...],
) -> bpy.types.Object:
    """Sculpt the palm, including its thenar fullness and finger-root arc."""
    if not (len(profile) == len(y_centers) == len(z_centers) == len(palmar_fullness) == len(thumb_reach)):
        raise ValueError("Each palm profile sequence must have the same length.")

    rings = []
    last_ring = len(profile) - 1
    for ring_index, ((x, radius_y, radius_z), y_center, z_center, fullness, reach) in enumerate(
        zip(
            profile,
            y_centers,
            z_centers,
            palmar_fullness,
            thumb_reach,
            strict=True,
        )
    ):
        ring = []
        for index in range(48):
            y, z = _superellipse(
                math.tau * index / 48,
                radius_y,
                radius_z,
                0.78,
            )
            palmar_weight = max(0.0, -y / radius_y)
            thenar_weight = 0.45 + 0.55 * max(0.0, -z / radius_z)
            y -= fullness * palmar_weight**1.5 * thenar_weight
            thumb_weight = max(0.0, -z / radius_z)
            z -= reach * thumb_weight**1.5 * (0.65 + 0.35 * palmar_weight)
            surface_z = z_center + z
            surface_x = _smooth_contour(distal_contour, surface_z) if ring_index == last_ring else x
            ring.append((side * surface_x, y_center + y, surface_z))
        rings.append(ring)
    return _loft_mesh(
        name,
        joint_index,
        rings,
        material,
        bevel=0.0015,
    )


def _smooth_contour(points: tuple[tuple[float, float], ...], coordinate: float) -> float:
    if coordinate <= points[0][0]:
        return points[0][1]
    for (start, start_value), (end, end_value) in pairwise(points):
        if coordinate <= end:
            weight = (coordinate - start) / (end - start)
            weight = 0.5 - 0.5 * math.cos(math.pi * weight)
            return start_value + weight * (end_value - start_value)
    return points[-1][1]


def _signed_power(value: float, exponent: float) -> float:
    return math.copysign(abs(value) ** exponent, value)


def _superellipse(angle: float, radius_a: float, radius_b: float, exponent: float) -> tuple[float, float]:
    return (
        radius_a * _signed_power(math.cos(angle), exponent),
        radius_b * _signed_power(math.sin(angle), exponent),
    )


def _loft_mesh(
    name: str,
    joint_index: int,
    rings: list[list[tuple[float, float, float]]],
    material: bpy.types.Material,
    *,
    bevel: float = 0.004,
) -> bpy.types.Object:
    vertices = [vertex for ring in rings for vertex in ring]
    segments = len(rings[0])
    faces = []
    for ring_index in range(len(rings) - 1):
        start = ring_index * segments
        faces.extend(_bridge_rings(start, start + segments, segments))
    faces.append(tuple(reversed(range(segments))))
    last_ring = (len(rings) - 1) * segments
    faces.append(tuple(last_ring + index for index in range(segments)))
    return _mesh_object(name, joint_index, vertices, faces, material, bevel=bevel)


def _super_loft_z(
    name: str,
    joint_index: int,
    profile: list[tuple[float, float, float]],
    material: bpy.types.Material,
    *,
    exponent: float = 0.55,
    bevel: float = 0.004,
) -> bpy.types.Object:
    rings = []
    for z, radius_x, radius_y in profile:
        ring = []
        for index in range(64):
            x, y = _superellipse(math.tau * index / 64, radius_x, radius_y, exponent)
            ring.append((x, y, z))
        rings.append(ring)
    return _loft_mesh(name, joint_index, rings, material, bevel=bevel)


def _super_loft_z_offset(
    name: str,
    joint_index: int,
    side: float,
    profile: list[tuple[float, float, float, float]],
    material: bpy.types.Material,
    *,
    exponent: float = 0.58,
) -> bpy.types.Object:
    rings = []
    for z, radius_x, radius_y, center_x in profile:
        ring = []
        for index in range(48):
            x, y = _superellipse(math.tau * index / 48, radius_x, radius_y, exponent)
            ring.append((side * center_x + x, y, z))
        rings.append(ring)
    return _loft_mesh(name, joint_index, rings, material)


def _super_loft_x(
    name: str,
    joint_index: int,
    side: float,
    profile: list[tuple[float, float, float]],
    material: bpy.types.Material,
    *,
    z_center: float = 0.0,
    exponent: float = 0.58,
    y_centers: tuple[float, ...] | None = None,
    z_centers: tuple[float, ...] | None = None,
) -> bpy.types.Object:
    if y_centers is None:
        y_centers = (0.0,) * len(profile)
    if z_centers is None:
        z_centers = (z_center,) * len(profile)
    if len(y_centers) != len(profile) or len(z_centers) != len(profile):
        raise ValueError("Each center sequence must match the profile.")
    rings = []
    for (x, radius_y, radius_z), y_center, ring_z in zip(
        profile,
        y_centers,
        z_centers,
        strict=True,
    ):
        ring = []
        for index in range(48):
            y, z = _superellipse(math.tau * index / 48, radius_y, radius_z, exponent)
            ring.append((side * x, y_center + y, ring_z + z))
        rings.append(ring)
    return _loft_mesh(name, joint_index, rings, material)


def _socketed_super_loft_x(
    name: str,
    joint_index: int,
    side: float,
    profile: list[tuple[float, float, float]],
    material: bpy.types.Material,
    *,
    socket_center: tuple[float, float, float],
    socket_radius: float,
    y_centers: tuple[float, ...],
    z_centers: tuple[float, ...],
    exponent: float = 0.58,
) -> bpy.types.Object:
    """Build a clean spherical cup into the proximal end of an arm loft."""
    segments = 48
    outer_rings = []
    for (x, radius_y, radius_z), y_center, z_center in zip(
        profile,
        y_centers,
        z_centers,
        strict=True,
    ):
        ring = []
        for index in range(segments):
            y, z = _superellipse(math.tau * index / segments, radius_y, radius_z, exponent)
            ring.append((side * x, y_center + y, z_center + z))
        outer_rings.append(ring)

    opening_x = side * profile[0][0]
    axial_offset = side * (opening_x - socket_center[0])
    opening_angle = math.acos(axial_offset / socket_radius)
    cavity_rings = []
    for step in range(8):
        angle = opening_angle * (1.0 - step / 8.0)
        axial = socket_radius * math.cos(angle)
        radial = socket_radius * math.sin(angle)
        cavity_rings.append(
            [
                (
                    socket_center[0] + side * axial,
                    socket_center[1] + radial * math.cos(math.tau * index / segments),
                    socket_center[2] + radial * math.sin(math.tau * index / segments),
                )
                for index in range(segments)
            ]
        )

    rings = [*outer_rings, *cavity_rings]
    vertices = [vertex for ring in rings for vertex in ring]
    pole = len(vertices)
    vertices.append(
        (
            socket_center[0] + side * socket_radius,
            socket_center[1],
            socket_center[2],
        )
    )
    faces = []
    for ring_index in range(len(outer_rings) - 1):
        start = ring_index * segments
        faces.extend(_bridge_rings(start, start + segments, segments))

    outer_end = (len(outer_rings) - 1) * segments
    faces.append(tuple(outer_end + index for index in range(segments)))
    cavity_start = len(outer_rings) * segments
    faces.extend(_bridge_rings(0, cavity_start, segments, reverse=True))
    for ring_index in range(len(cavity_rings) - 1):
        start = cavity_start + ring_index * segments
        faces.extend(_bridge_rings(start, start + segments, segments, reverse=True))
    last_cavity = cavity_start + (len(cavity_rings) - 1) * segments
    for index in range(segments):
        faces.append((last_cavity + index, pole, last_cavity + (index + 1) % segments))

    if side < 0.0:
        faces = [tuple(reversed(face)) for face in faces]
    return _mesh_object(name, joint_index, vertices, faces, material, bevel=0.0015)


def _bridge_rings(
    start: int,
    end: int,
    segments: int,
    *,
    reverse: bool = False,
) -> list[tuple[int, ...]]:
    faces = [
        (
            start + index,
            start + (index + 1) % segments,
            end + (index + 1) % segments,
            end + index,
        )
        for index in range(segments)
    ]
    return [tuple(reversed(face)) for face in faces] if reverse else faces


def _super_loft_y(
    name: str,
    joint_index: int,
    side: float,
    profile: list[tuple[float, float, float, float]],
    material: bpy.types.Material,
    *,
    exponent: float = 0.58,
    x_center: float = 0.103,
) -> bpy.types.Object:
    rings = []
    for y, radius_x, radius_z, z_center in profile:
        ring = []
        for index in range(48):
            x, z = _superellipse(math.tau * index / 48, radius_x, radius_z, exponent)
            ring.append((side * x_center + x, y, z_center + z))
        rings.append(ring)
    return _loft_mesh(name, joint_index, rings, material)


def _carve_sockets(
    obj: bpy.types.Object,
    sockets: tuple[tuple[tuple[float, float, float], float], ...],
    *,
    rim_width: float = 0.002,
    symmetrize: bool = False,
) -> None:
    cutters = []
    for center, radius in sockets:
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=64,
            ring_count=32,
            radius=radius + SOCKET_CLEARANCE,
            location=center,
        )
        cutters.append(bpy.context.object)
    bpy.ops.object.select_all(action="DESELECT")
    for cutter in cutters:
        cutter.select_set(True)
    bpy.context.view_layer.objects.active = cutters[0]
    if len(cutters) > 1:
        bpy.ops.object.join()
    cutter = cutters[0]
    cutter.name = f"{obj.name}__socket_cutters"

    bpy.context.view_layer.objects.active = obj
    modifier = obj.modifiers.new(name="Ball sockets", type="BOOLEAN")
    modifier.operation = "DIFFERENCE"
    modifier.solver = "EXACT"
    modifier.object = cutter
    bpy.ops.object.modifier_apply(modifier=modifier.name)

    bpy.data.objects.remove(cutter, do_unlink=True)
    bpy.context.view_layer.objects.active = obj
    bevel = obj.modifiers.new(name="Rounded socket rims", type="BEVEL")
    bevel.width = rim_width
    bevel.segments = 6
    bevel.limit_method = "ANGLE"
    bevel.angle_limit = math.radians(20.0)
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    if symmetrize:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.symmetrize(direction="NEGATIVE_X", threshold=1e-6)
        bpy.ops.object.mode_set(mode="OBJECT")
    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def _ellipsoid(
    name: str,
    joint_index: int,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    segments: int = 64,
    rings: int = 32,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name
    obj.scale = radii
    obj.data.materials.append(material)
    obj["smpl_joint_index"] = joint_index
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def _rounded_disk(
    name: str,
    joint_index: int,
    center: tuple[float, float, float],
    radii: tuple[float, float],
    height: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """Create a short elliptical cylinder with a softly rolled rim."""
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=64,
        radius=1.0,
        depth=height,
        end_fill_type="NGON",
        location=center,
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name
    obj.scale = (radii[0], radii[1], 1.0)
    obj.data.materials.append(material)
    obj["smpl_joint_index"] = joint_index
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bevel = obj.modifiers.new(name="Rounded rim", type="BEVEL")
    bevel.width = min(height * 0.28, radii[0] * 0.12, radii[1] * 0.12)
    bevel.segments = 5
    bevel.limit_method = "ANGLE"
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def _sculpted_head(
    name: str,
    joint_index: int,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """Create a featureless head with a facial plane, cheeks, jaw, and cranium."""
    profile = (
        # z, half-width, front depth, back depth, y center
        (0.355, 0.031, 0.041, 0.036, -0.010),
        (0.368, 0.047, 0.050, 0.046, -0.008),
        (0.387, 0.063, 0.060, 0.058, -0.006),
        (0.415, 0.078, 0.070, 0.076, -0.004),
        (0.447, 0.091, 0.076, 0.092, -0.001),
        (0.483, 0.093, 0.074, 0.102, 0.002),
        (0.521, 0.091, 0.069, 0.108, 0.005),
        (0.559, 0.084, 0.061, 0.101, 0.007),
        (0.587, 0.067, 0.048, 0.079, 0.007),
        (0.602, 0.034, 0.029, 0.038, 0.005),
    )
    rings = []
    for z, radius_x, front_depth, back_depth, y_center in profile:
        ring = []
        for index in range(72):
            angle = math.tau * index / 72
            sine = math.sin(angle)
            depth = front_depth if sine < 0.0 else back_depth
            depth_exponent = 0.48 if sine < 0.0 else 0.82
            x = radius_x * _signed_power(math.cos(angle), 0.82)
            y = y_center + depth * _signed_power(sine, depth_exponent)
            if y < 0.0 and z < 0.420:
                height_weight = 1.0 - (z - 0.355) / 0.065
                center_weight = max(0.0, 1.0 - abs(x) / 0.075)
                y -= 0.010 * height_weight * center_weight
            ring.append((x, y, z))
        rings.append(ring)
    obj = _loft_mesh(name, joint_index, rings, material, bevel=0.0)
    subdivision = obj.modifiers.new("Sculpt smoothing", "SUBSURF")
    subdivision.levels = 1
    subdivision.render_levels = 1
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    with bpy.context.temp_override(
        object=obj,
        active_object=obj,
        selected_objects=[obj],
        selected_editable_objects=[obj],
    ):
        bpy.ops.object.modifier_apply(modifier=subdivision.name)
    bevel = obj.modifiers.new("Polished transitions", "BEVEL")
    bevel.width = 0.0025
    bevel.segments = 3
    bevel.limit_method = "ANGLE"
    with bpy.context.temp_override(
        object=obj,
        active_object=obj,
        selected_objects=[obj],
        selected_editable_objects=[obj],
    ):
        bpy.ops.object.modifier_apply(modifier=bevel.name)
    return obj


def _inset_segment_endpoint(
    endpoint: tuple[float, float, float],
    toward: tuple[float, float, float],
    distance: float,
) -> tuple[float, float, float]:
    endpoint_vector = Vector(endpoint)
    direction = Vector(toward) - endpoint_vector
    return tuple(endpoint_vector + distance * direction.normalized())


def _rounded_tapered_segment(
    name: str,
    joint_index: int,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    start_radius: float,
    end_radius: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """Create a polished rigid phalanx with a gentle longitudinal taper."""
    start_vector = Vector(start)
    end_vector = Vector(end)
    direction = end_vector - start_vector
    bpy.ops.mesh.primitive_cone_add(
        vertices=48,
        radius1=start_radius,
        radius2=end_radius,
        depth=direction.length,
        end_fill_type="NGON",
        location=(start_vector + end_vector) / 2,
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(direction.normalized())
    obj.data.materials.append(material)
    obj["smpl_joint_index"] = joint_index
    bpy.context.view_layer.objects.active = obj

    bevel = obj.modifiers.new(name="Soft phalanx rims", type="BEVEL")
    bevel.width = min(start_radius, end_radius) * 0.28
    bevel.segments = 4
    bevel.limit_method = "ANGLE"
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(collection):
            collection.remove(block)


def _material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    metallic: float,
    roughness: float,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return material


def _mesh_object(
    name: str,
    joint_index: int,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, ...]],
    material: bpy.types.Material,
    *,
    bevel: float,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    obj["smpl_joint_index"] = joint_index
    if bevel:
        modifier = obj.modifiers.new("Soft edge", "BEVEL")
        modifier.width = bevel
        modifier.segments = 4
        modifier.limit_method = "ANGLE"
        modifier.angle_limit = math.radians(25.0)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return obj


def _studio(floor_material: bpy.types.Material) -> None:
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, -1.18))
    floor = bpy.context.object
    floor.name = "STUDIO_FLOOR"
    floor.data.materials.append(floor_material)

    world = bpy.context.scene.world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.055, 0.052, 0.048, 1.0)
    background.inputs["Strength"].default_value = 0.32
    target = Vector((0.0, 0.0, -0.28))
    for name, energy, size, location, color in (
        ("Key", 1050.0, 3.2, (-2.7, -3.2, 3.2), (1.0, 0.91, 0.84)),
        ("Fill", 650.0, 3.0, (2.8, -2.0, 1.4), (0.78, 0.86, 1.0)),
        ("Rim", 900.0, 2.5, (0.0, 2.8, 2.8), (1.0, 0.82, 0.76)),
        ("Top", 800.0, 2.2, (0.0, 0.0, 4.0), (1.0, 0.96, 0.90)),
    ):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        data.color = color
        light = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (target - light.location).to_track_quat("-Z", "Y").to_euler()


def _add_camera() -> bpy.types.Object:
    data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", data)
    bpy.context.collection.objects.link(camera)
    data.sensor_width = 36
    bpy.context.scene.camera = camera
    return camera


def _configure_render() -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"


def _render(
    camera: bpy.types.Object,
    output: Path,
    location: tuple[float, float, float],
    lens: float,
    *,
    target: tuple[float, float, float] = (0.0, 0.0, -0.32),
) -> None:
    camera.location = location
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = lens
    bpy.context.scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def _export_character(path: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and obj.name != "STUDIO_FLOOR":
            obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_extras=True,
        export_materials="EXPORT",
    )
    bpy.ops.object.select_all(action="DESELECT")


def _character_mesh_count() -> int:
    return sum(obj.type == "MESH" and obj.name != "STUDIO_FLOOR" for obj in bpy.context.scene.objects)


if __name__ == "__main__":
    main()
