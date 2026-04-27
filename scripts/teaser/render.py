#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.13.*"  # bpy>=5.1 ships cp313 wheels built against numpy 2.x
# dependencies = [
#   "body-models",
#   "bpy>=5.1.0",
# ]
# [tool.uv.sources]
# body-models = { path = "../.." }
# ///
"""Render the README body-model lineup directly from the body-models API.

Self-contained PEP 723 script: loads each body model in-process, builds a
canonical T-pose mesh, and renders the lineup with Cycles via headless ``bpy``.

Usage:
    uv run scripts/teaser/render.py [--output PATH] [--samples N] [--denoise]

A neighbouring ``.python-version`` pins this script to Python 3.13 (required
by ``bpy>=5.1``); the repo-root pin of 3.12 stays in effect everywhere else.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from body_models.anny.numpy import ANNY
from body_models.flame.numpy import FLAME
from body_models.g1.numpy import G1
from body_models.garment_measurements.numpy import GarmentMeasurements
from body_models.mhr.numpy import MHR
from body_models.skel.numpy import SKEL
from body_models.smpl.numpy import SMPL
from body_models.smplx.numpy import SMPLX
from body_models.soma.numpy import SOMA

# ── Lineup configuration ─────────────────────────────────────────────────────
ASSETS_DIR = Path(__file__).parent.parent.parent / "tests" / "assets"
MODEL_HEIGHT = 1.75
MODEL_GAP = 0.30

# Insertion order doubles as the canonical lineup ordering.
PASTELS = {
    "smpl": (0.95, 0.63, 0.72, 1.0),  # rose
    "smplx": (0.62, 0.78, 0.98, 1.0),  # sky
    "skel": (0.62, 0.93, 0.74, 1.0),  # mint
    "mhr": (0.99, 0.73, 0.54, 1.0),  # peach
    "anny": (0.76, 0.68, 0.98, 1.0),  # lavender
    "flame": (0.99, 0.93, 0.62, 1.0),  # butter
    "garment_measurements": (0.69, 0.86, 0.93, 1.0),  # powder
    "soma": (0.97, 0.78, 0.78, 1.0),  # coral
    "g1": (0.78, 0.78, 0.86, 1.0),  # steel
}
LABELS = {f: f.upper() for f in PASTELS} | {"garment_measurements": "GARMENT\nMEASUREMENTS"}
# FLAME is head-only: half-size keeps it in scale with the row.
SCALES = {"flame": 0.5}

LOADERS = {
    "smpl": lambda: SMPL(gender="neutral"),  # path via body-models config
    "smplx": lambda: SMPLX(gender="neutral"),  # path via body-models config
    "skel": lambda: SKEL(ASSETS_DIR / "skel/model", "male"),
    "mhr": lambda: MHR(ASSETS_DIR / "mhr/model"),
    "anny": lambda: ANNY(ASSETS_DIR / "anny/model"),
    "flame": lambda: FLAME(ASSETS_DIR / "flame/model/FLAME_NEUTRAL.pkl"),
    "garment_measurements": lambda: GarmentMeasurements(ASSETS_DIR / "garment_measurements/model"),
    "soma": lambda: SOMA(),
    "g1": lambda: G1(ASSETS_DIR / "g1/model", rotation_type="hinge"),
}

# ── Pose adapters: bring rest poses to a uniform T-pose ──────────────────────
# (joint_name_lowercase, axis) → axis-angle radians.
ANNY_POSE_OFFSETS = {
    ("shoulder01.l", 2): 0.475,
    ("shoulder01.r", 2): -0.475,
    ("upperarm01.l", 2): 0.6,
    ("upperarm01.r", 2): -0.6,
    ("clavicle.l", 2): -0.225,
    ("clavicle.r", 2): 0.225,
    ("upperleg01.l", 2): 0.09,
    ("upperleg01.r", 2): -0.09,
    ("lowerarm01.l", 0): -0.75,
    ("lowerarm01.r", 0): -0.75,
}
ANNY_GLOBAL_PITCH_X = 0.08

GM_POSE_OFFSETS = {
    ("upper_arm_l", 2): 0.6,
    ("upper_arm_r", 2): -0.6,
    ("clavicle_l", 2): 0.15,
    ("clavicle_r", 2): -0.15,
    ("thigh_l", 2): 0.12,
    ("thigh_r", 2): -0.12,
}

# G1 hinge body_pose: scalar per qpos joint. 16/23 = shoulder roll abducts the
# arms; 18/25 = elbow extension straightens the forearms laterally.
G1_HINGE_OFFSETS = {16: np.pi / 2, 23: -np.pi / 2, 18: 1.0, 25: 1.0}

# MHR: rows of [tx, ty, tz, euler_x, euler_y, euler_z, scale] per joint.
MHR_TARGETS = (
    ("l_uparm", 4, 0.8),
    ("r_uparm", 4, 0.8),
    ("l_lowarm", 4, -0.4),
    ("r_lowarm", 4, -0.4),
    ("l_lowarm", 5, -0.6),
    ("r_lowarm", 5, -0.6),
    ("l_upleg", 4, 0.12),
    ("r_upleg", 4, 0.12),
    ("l_upleg", 5, -0.06),
    ("r_upleg", 5, -0.06),
)


# ── Top-level pipeline ───────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("[1/5] Generating meshes...", flush=True)
    meshes = [(family, *canonical_mesh(family)) for family in PASTELS]

    print("[2/5] Building scene...", flush=True)
    clear_scene()
    objects = instantiate_lineup(meshes)
    add_lights()
    set_world_background()

    print("[3/5] Camera & labels...", flush=True)
    camera = set_camera(objects)
    labels = add_family_labels([m[0] for m in meshes], objects, camera)
    set_render_border(camera, objects + labels)

    print("[4/5] Configuring renderer...", flush=True)
    configure_render(args)

    print("[5/5] Rendering...", flush=True)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render the README body-model lineup")
    p.add_argument("--output", type=Path, default=Path("README_render.png"))
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--width", type=int, default=2200)
    p.add_argument("--height", type=int, default=1200)
    p.add_argument("--denoise", action="store_true")
    return p.parse_args()


# ── Per-family canonical mesh ────────────────────────────────────────────────
def canonical_mesh(family: str) -> tuple[np.ndarray, np.ndarray]:
    model = LOADERS[family]()
    if family in ("smpl", "smplx", "skel", "flame"):
        verts = np.asarray(model.rest_vertices, dtype=np.float32)
    else:
        params = model.get_rest_pose(batch_size=1)
        if family == "anny":
            apply_pose_offsets(model, params, ANNY_POSE_OFFSETS)
            params["global_rotation"][0, 0] = ANNY_GLOBAL_PITCH_X
        elif family == "mhr":
            params["pose"][0] = solve_mhr_pose(model)
        elif family == "garment_measurements":
            apply_pose_offsets(model, params, GM_POSE_OFFSETS)
        elif family == "g1":
            for qpos_idx, value in G1_HINGE_OFFSETS.items():
                params["body_pose"][0, qpos_idx, 0] = value
        verts = np.asarray(model.forward_vertices(**params)[0], dtype=np.float32)
    return verts, np.asarray(model.faces, dtype=np.int32)


def joint_index(model, name: str) -> int:
    for i, jn in enumerate(model.joint_names):
        if jn.lower() == name:
            return i
    raise ValueError(f"Joint not found in model: {name!r}")


def apply_pose_offsets(model, params, offsets):
    for (joint_name, axis), value in offsets.items():
        params["pose"][0, joint_index(model, joint_name), axis] = value


def solve_mhr_pose(model) -> np.ndarray:
    """Least-squares solve for the MHR pose vector that hits the target Euler offsets."""
    rows = [joint_index(model, j) * 7 + c for j, c, _ in MHR_TARGETS]
    targets = np.asarray([t for _, _, t in MHR_TARGETS], dtype=np.float64)
    transform = np.asarray(model.parameter_transform, dtype=np.float64)
    system = transform[np.asarray(rows), : model.pose_dim]
    x, *_ = np.linalg.lstsq(system, targets, rcond=1e-4)
    return x.astype(np.float32)


# ── Scene construction ──────────────────────────────────────────────────────
def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for blocks in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.curves,
    ):
        for block in list(blocks):
            if block.users == 0:
                blocks.remove(block)


def instantiate_lineup(meshes):
    normalized = [(name, normalize_mesh(v, MODEL_HEIGHT * SCALES.get(name, 1.0)), f) for name, v, f in meshes]
    widths = [float(v[:, 0].max() - v[:, 0].min()) for _, v, _ in normalized]

    objects: list[bpy.types.Object] = []
    xpos = -0.5 * (sum(widths) + MODEL_GAP * (len(widths) - 1))
    for (name, vertices, faces), width in zip(normalized, widths):
        obj = create_mesh_object(name, vertices, faces)
        obj.location.x = xpos + 0.5 * width
        obj.rotation_euler[2] = np.pi  # face the camera at -Y
        obj.data.materials.append(build_material(f"{name}_Material", PASTELS[name]))
        xpos += width + MODEL_GAP
        objects.append(obj)
    bpy.context.view_layer.update()
    return objects


def normalize_mesh(vertices: np.ndarray, target_height: float) -> np.ndarray:
    norm = vertices.copy().astype(np.float32)
    norm[:, 0] -= 0.5 * (float(norm[:, 0].min()) + float(norm[:, 0].max()))
    norm[:, 1] -= float(norm[:, 1].min())
    height = float(norm[:, 1].max() - norm[:, 1].min())
    if height > 1e-6:
        norm *= target_height / height
    return norm


def create_mesh_object(name, vertices, faces):
    vertices = vertices.copy()
    vertices[:, 1] -= float(vertices[:, 1].min())
    # Body models are Y-up, Blender is Z-up.
    verts_blender = [(float(x), float(z), float(y)) for x, y, z in vertices]

    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(verts_blender, [], [tuple(map(int, p)) for p in faces])
    mesh.update(calc_edges=True)

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    bpy.ops.object.shade_auto_smooth(use_auto_smooth=True, angle=np.radians(30.0))
    return obj


def build_material(name, color):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.45
    bsdf.inputs["Specular IOR Level"].default_value = 0.18
    return mat


def add_lights() -> None:
    add_sun("KeyLight", location=(3.0, -4.0, 5.5), rotation_deg=(52, 0, 35), energy=2.4)
    add_sun("FillLight", location=(-3.0, 2.0, 3.0), rotation_deg=(45, 0, -120), energy=0.9, color=(0.70, 0.78, 1.0))


def add_sun(name, *, location, rotation_deg, energy, color=(1.0, 1.0, 1.0)):
    data = bpy.data.lights.new(name=name, type="SUN")
    data.energy = energy
    data.color = color
    obj = bpy.data.objects.new(name=name, object_data=data)
    obj.location = location
    obj.rotation_euler = tuple(np.radians(d) for d in rotation_deg)
    bpy.context.scene.collection.objects.link(obj)


def set_world_background() -> None:
    world = bpy.data.worlds[0] if bpy.data.worlds else bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.02, 0.02, 0.06, 1.0)
    bg.inputs["Strength"].default_value = 0.25


def set_camera(model_objects):
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_data.lens = 38.0
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    bounds = np.asarray(
        [np.asarray(obj.matrix_world @ Vector(c), dtype=np.float32) for obj in model_objects for c in obj.bound_box]
    )
    mins, maxs = bounds.min(axis=0), bounds.max(axis=0)
    center = 0.5 * (mins + maxs)
    height = float(maxs[2] - mins[2])
    distance = max(5.0, max(float(maxs[0] - mins[0]), height) * 1.05 + 0.9)
    cam.location = (float(center[0]), float(center[1]) - distance, float(mins[2] + 0.70 * height))

    target = bpy.data.objects.new("CameraTarget", None)
    target.empty_display_type = "PLAIN_AXES"
    target.location = (float(center[0]), float(center[1]), float(mins[2] + 0.62 * height))
    bpy.context.scene.collection.objects.link(target)

    track_to(cam, target, "TRACK_NEGATIVE_Z", "UP_Y")
    return cam


def add_family_labels(names, model_objects, camera):
    labels = []
    for name, obj in zip(names, model_objects):
        mins, maxs = object_bounds(obj)
        center = 0.5 * (mins + maxs)

        curve = bpy.data.curves.new(name=f"{name}_LabelCurve", type="FONT")
        curve.body = LABELS[name]
        curve.align_x = "CENTER"
        curve.align_y = "CENTER"
        curve.extrude = 0.01
        curve.bevel_depth = 0.002
        curve.size = 0.22

        label = bpy.data.objects.new(f"{name}_Label", curve)
        bpy.context.scene.collection.objects.link(label)
        label.location = (float(center[0]), float(center[1]), float(maxs[2] + 0.22))

        track_to(label, camera, "TRACK_Z", "UP_Y")
        label.data.materials.append(build_material(f"{name}_LabelMaterial", (0.93, 0.94, 0.95, 1.0)))
        labels.append(label)
    return labels


def track_to(source, target, track_axis, up_axis):
    constraint = source.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = track_axis
    constraint.up_axis = up_axis


def object_bounds(obj):
    corners = np.asarray([np.asarray(obj.matrix_world @ Vector(c), dtype=np.float32) for c in obj.bound_box])
    return np.min(corners, axis=0), np.max(corners, axis=0)


def set_render_border(camera, objects):
    pad = 0.03
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    min_x, min_y, max_x, max_y = 1.0, 1.0, 0.0, 0.0
    for obj in objects:
        if obj.type == "MESH":
            points = (obj.matrix_world @ v.co for v in obj.data.vertices)
        else:
            points = (obj.matrix_world @ Vector(c) for c in obj.bound_box)
        for co in points:
            ndc = world_to_camera_view(scene, camera, co)
            min_x, min_y = min(min_x, float(ndc.x)), min(min_y, float(ndc.y))
            max_x, max_y = max(max_x, float(ndc.x)), max(max_y, float(ndc.y))
    scene.render.use_border = True
    scene.render.use_crop_to_border = True
    scene.render.border_min_x = max(0.0, min_x - pad)
    scene.render.border_min_y = max(0.0, min_y - pad)
    scene.render.border_max_x = min(1.0, max_x + pad)
    scene.render.border_max_y = min(1.0, max_y + pad)


def configure_render(args):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.filepath = str(args.output)
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.2
    scene.view_settings.gamma = 1.0

    cy = scene.cycles
    cy.samples = args.samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.003
    cy.max_bounces = 14
    cy.diffuse_bounces = 6
    cy.glossy_bounces = 6
    cy.transmission_bounces = 8
    cy.transparent_max_bounces = 16
    cy.volume_bounces = 2
    cy.caustics_reflective = True
    cy.caustics_refractive = True
    cy.use_denoising = args.denoise


if __name__ == "__main__":
    main()
