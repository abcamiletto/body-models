#!/usr/bin/env python3
"""Render the README body-model lineup in Blender from local OBJ meshes.

Pipeline:
1) Export canonical family meshes via scripts/teaser/export_readme_meshes.py (uv env)
2) Load local OBJ files in Blender and render lineup
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import bpy
from mathutils import Vector
import numpy as np

FAMILY_ORDER = ("smpl", "smplx", "skel", "mhr", "anny")
PASTELS = (
    (0.95, 0.63, 0.72, 1.0),  # rose
    (0.62, 0.78, 0.98, 1.0),  # sky
    (0.62, 0.93, 0.74, 1.0),  # mint
    (0.99, 0.73, 0.54, 1.0),  # peach
    (0.76, 0.68, 0.98, 1.0),  # lavender
)
AUTO_SMOOTH_ANGLE = np.radians(30.0)
MODEL_HEIGHT = 1.75
MODEL_GAP = 0.30


def blender_argv() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render body-model lineup image")
    p.add_argument("--output", type=Path, default=Path("README_render.png"), help="Output PNG path")
    p.add_argument("--mesh-dir", type=Path, default=Path("scripts/teaser/.meshes"), help="Directory with OBJ meshes")
    p.add_argument("--refresh-meshes", action="store_true", help="Force re-export OBJ meshes")
    p.add_argument("--skip-export", action="store_true", help="Do not run exporter")
    p.add_argument("--samples", type=int, default=512, help="Cycles samples")
    p.add_argument("--engine", choices=["CYCLES", "BLENDER_EEVEE"], default="CYCLES")
    p.add_argument("--max-bounces", type=int, default=14, help="Cycles max light bounces")
    p.add_argument("--diffuse-bounces", type=int, default=6, help="Cycles diffuse bounces")
    p.add_argument("--glossy-bounces", type=int, default=6, help="Cycles glossy bounces")
    p.add_argument("--transmission-bounces", type=int, default=8, help="Cycles transmission bounces")
    p.add_argument("--transparent-max-bounces", type=int, default=16, help="Cycles transparent bounces")
    p.add_argument("--volume-bounces", type=int, default=2, help="Cycles volume bounces")
    p.add_argument("--adaptive-threshold", type=float, default=0.003, help="Cycles adaptive sampling threshold")
    p.add_argument("--denoise", action="store_true", help="Enable Cycles denoising")
    p.add_argument("--width", type=int, default=2200, help="Output width in px")
    p.add_argument("--height", type=int, default=1200, help="Output height in px")
    p.add_argument("--title", type=str, default="", help="Optional global title text")
    p.add_argument(
        "--transparent-bg",
        dest="transparent_bg",
        action="store_true",
        default=True,
        help="Render with transparent background (default: on)",
    )
    p.add_argument(
        "--opaque-bg",
        dest="transparent_bg",
        action="store_false",
        help="Render with opaque background and ground plane",
    )
    return p.parse_args(blender_argv())


def parse_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            tokens = line.split()
            if len(tokens) >= 4:
                vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            continue
        if line.startswith("f "):
            tokens = line.split()[1:]
            indices = [int(token.split("/")[0]) - 1 for token in tokens]
            if len(indices) < 3:
                continue
            for i in range(1, len(indices) - 1):
                faces.append([indices[0], indices[i], indices[i + 1]])

    if not vertices or not faces:
        raise ValueError(f"Could not parse OBJ mesh from {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def ensure_mesh_exports(mesh_dir: Path, refresh: bool, skip_export: bool) -> None:
    mesh_dir.mkdir(parents=True, exist_ok=True)
    missing = [family for family in FAMILY_ORDER if not (mesh_dir / f"{family}.obj").exists()]
    if skip_export or (not refresh and not missing):
        return

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    exporter = script_path.with_name("export_readme_meshes.py")
    cmd = ["uv", "run", "python", str(exporter), "--out", str(mesh_dir)]
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    if proc.returncode != 0:
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        raise RuntimeError(f"Mesh export failed:\n{output.strip()}")


def load_required_meshes(mesh_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    missing = [family for family in FAMILY_ORDER if not (mesh_dir / f"{family}.obj").exists()]
    if missing:
        missing_joined = ", ".join(missing)
        raise RuntimeError(f"Missing required OBJ meshes in {mesh_dir}: {missing_joined}")

    loaded: list[tuple[str, np.ndarray, np.ndarray]] = []
    for family in FAMILY_ORDER:
        vertices, faces = parse_obj_mesh(mesh_dir / f"{family}.obj")
        loaded.append((family, vertices, faces))
    return loaded


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for block_collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.curves,
    ):
        for block in list(block_collection):
            if block.users == 0:
                block_collection.remove(block)


def normalize_mesh(vertices: np.ndarray, target_height: float = MODEL_HEIGHT) -> np.ndarray:
    normalized = vertices.copy().astype(np.float32)
    normalized[:, 0] -= 0.5 * (float(normalized[:, 0].min()) + float(normalized[:, 0].max()))
    normalized[:, 1] -= float(normalized[:, 1].min())
    height = float(normalized[:, 1].max() - normalized[:, 1].min())
    if height > 1e-6:
        normalized *= target_height / height
    return normalized


def create_mesh_object(vertices: np.ndarray, faces: np.ndarray, name: str) -> bpy.types.Object:
    vertices = vertices.copy()
    vertices[:, 1] -= float(vertices[:, 1].min())

    # Body-models are Y-up; Blender is Z-up.
    verts_blender = [(float(x), float(z), float(y)) for x, y, z in vertices]
    faces_py = [tuple(map(int, tri)) for tri in faces]

    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(verts_blender, [], faces_py)
    mesh.update(calc_edges=True)

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    bpy.ops.object.shade_auto_smooth(use_auto_smooth=True, angle=AUTO_SMOOTH_ANGLE)
    return obj


def build_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    out = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.45
    bsdf.inputs["Specular IOR Level"].default_value = 0.18
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def object_bounds(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    world_corners = np.asarray([np.array(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box], dtype=np.float32)
    return np.min(world_corners, axis=0), np.max(world_corners, axis=0)


def scene_bounds(objects: list[bpy.types.Object]) -> tuple[np.ndarray, np.ndarray]:
    mins = np.array([1e9, 1e9, 1e9], dtype=np.float32)
    maxs = np.array([-1e9, -1e9, -1e9], dtype=np.float32)
    for obj in objects:
        obj_min, obj_max = object_bounds(obj)
        mins = np.minimum(mins, obj_min)
        maxs = np.maximum(maxs, obj_max)
    return mins, maxs


def add_track_to_constraint(
    obj: bpy.types.Object,
    target: bpy.types.Object,
    track_axis: str,
    up_axis: str = "UP_Y",
) -> None:
    constraint = obj.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = track_axis
    constraint.up_axis = up_axis


def add_ground(size: float) -> None:
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, 0.0))
    ground = bpy.context.active_object
    mat = build_material("GroundMaterial", (0.09, 0.10, 0.17, 1.0))
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.95
    bsdf.inputs["Metallic"].default_value = 0.0
    ground.data.materials.append(mat)


def add_lights() -> None:
    key_data = bpy.data.lights.new(name="KeyLight", type="SUN")
    key_data.energy = 2.4
    key = bpy.data.objects.new(name="KeyLight", object_data=key_data)
    key.location = (3.0, -4.0, 5.5)
    key.rotation_euler = (np.radians(52), 0.0, np.radians(35))
    bpy.context.scene.collection.objects.link(key)

    fill_data = bpy.data.lights.new(name="FillLight", type="SUN")
    fill_data.energy = 0.9
    fill_data.color = (0.70, 0.78, 1.0)
    fill = bpy.data.objects.new(name="FillLight", object_data=fill_data)
    fill.location = (-3.0, 2.0, 3.0)
    fill.rotation_euler = (np.radians(45), 0.0, np.radians(-120))
    bpy.context.scene.collection.objects.link(fill)


def set_world_background() -> None:
    world = bpy.data.worlds[0] if bpy.data.worlds else bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.02, 0.02, 0.06, 1.0)
    bg.inputs["Strength"].default_value = 0.25


def instantiate_lineup(meshes: list[tuple[str, np.ndarray, np.ndarray]]) -> list[bpy.types.Object]:
    normalized = [(name, normalize_mesh(vertices), faces) for name, vertices, faces in meshes]
    widths = [float(vertices[:, 0].max() - vertices[:, 0].min()) for _, vertices, _ in normalized]

    objects: list[bpy.types.Object] = []
    xpos = -0.5 * (sum(widths) + MODEL_GAP * (len(widths) - 1))
    for idx, (name, vertices, faces) in enumerate(normalized):
        width = widths[idx]
        obj = create_mesh_object(vertices, faces, name=name)
        obj.location.x = xpos + 0.5 * width
        obj.rotation_euler[2] = np.radians(180.0)
        obj.data.materials.append(build_material(f"{name}_Material", PASTELS[idx]))
        xpos += width + MODEL_GAP
        objects.append(obj)

    bpy.context.view_layer.update()
    return objects


def add_3d_title(title: str, model_objects: list[bpy.types.Object], camera: bpy.types.Object) -> None:
    if not title.strip():
        return
    mins, maxs = scene_bounds(model_objects)
    center = 0.5 * (mins + maxs)
    width = float(maxs[0] - mins[0])

    curve = bpy.data.curves.new(name="TitleCurve", type="FONT")
    curve.body = title
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.extrude = 0.03
    curve.bevel_depth = 0.004
    curve.size = 0.6

    text_obj = bpy.data.objects.new("TitleText", curve)
    bpy.context.scene.collection.objects.link(text_obj)
    front_y = float(mins[1] - 0.20) if camera.location.y < center[1] else float(maxs[1] + 0.20)
    text_obj.location = (float(center[0]), front_y, float(maxs[2] + 0.16))

    bpy.context.view_layer.update()
    target_width = max(2.0, 0.72 * width)
    if text_obj.dimensions.x > 1e-6:
        scale = target_width / text_obj.dimensions.x
        text_obj.scale = (scale, scale, scale)

    add_track_to_constraint(text_obj, camera, track_axis="TRACK_Z")
    text_obj.data.materials.append(build_material("TitleMaterial", (0.95, 0.94, 0.88, 1.0)))


def add_family_labels(names: list[str], model_objects: list[bpy.types.Object], camera: bpy.types.Object) -> None:
    for name, obj in zip(names, model_objects):
        mins, maxs = object_bounds(obj)
        center = 0.5 * (mins + maxs)

        curve = bpy.data.curves.new(name=f"{name}_LabelCurve", type="FONT")
        curve.body = name.upper()
        curve.align_x = "CENTER"
        curve.align_y = "CENTER"
        curve.extrude = 0.01
        curve.bevel_depth = 0.002
        curve.size = 0.22

        label = bpy.data.objects.new(f"{name}_Label", curve)
        bpy.context.scene.collection.objects.link(label)
        label.location = (float(center[0]), float(center[1]), float(maxs[2] + 0.22))

        add_track_to_constraint(label, camera, track_axis="TRACK_Z")
        label.data.materials.append(build_material(f"{name}_LabelMaterial", (0.93, 0.94, 0.95, 1.0)))


def set_camera(model_objects: list[bpy.types.Object]) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_data.lens = 38.0
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    mins, maxs = scene_bounds(model_objects)
    center = 0.5 * (mins + maxs)
    width = float(maxs[0] - mins[0])
    height = float(maxs[2] - mins[2])
    diag = max(width, height)

    distance = max(5.0, diag * 1.05 + 0.9)
    cam.location = (float(center[0]), float(center[1]) - distance, float(mins[2] + 0.70 * height))

    target = bpy.data.objects.new("CameraTarget", None)
    target.empty_display_type = "PLAIN_AXES"
    target.location = (float(center[0]), float(center[1]), float(mins[2] + 0.62 * height))
    bpy.context.scene.collection.objects.link(target)
    add_track_to_constraint(cam, target, track_axis="TRACK_NEGATIVE_Z")
    return cam


def configure_render(args: argparse.Namespace, output_path: Path) -> None:
    scene = bpy.context.scene
    scene.render.engine = args.engine
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = bool(args.transparent_bg)
    scene.render.filepath = str(output_path)
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.2
    scene.view_settings.gamma = 1.0

    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = args.adaptive_threshold
        scene.cycles.max_bounces = args.max_bounces
        scene.cycles.diffuse_bounces = args.diffuse_bounces
        scene.cycles.glossy_bounces = args.glossy_bounces
        scene.cycles.transmission_bounces = args.transmission_bounces
        scene.cycles.transparent_max_bounces = args.transparent_max_bounces
        scene.cycles.volume_bounces = args.volume_bounces
        scene.cycles.caustics_reflective = True
        scene.cycles.caustics_refractive = True
        scene.cycles.use_denoising = bool(args.denoise)
    else:
        scene.eevee.taa_render_samples = 64


def main() -> None:
    args = parse_args()
    mesh_dir = args.mesh_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[1/8] Ensuring local OBJ exports...", flush=True)
    ensure_mesh_exports(mesh_dir, args.refresh_meshes, args.skip_export)

    print("[2/8] Loading canonical family OBJs...", flush=True)
    loaded = load_required_meshes(mesh_dir)
    print(f"Loaded families: {[name for name, _, _ in loaded]}", flush=True)

    print("[3/8] Building scene...", flush=True)
    clear_scene()
    model_objects = instantiate_lineup(loaded)

    print("[4/8] Adding environment...", flush=True)
    add_lights()
    set_world_background()
    if args.transparent_bg:
        print("Transparent background enabled; skipping ground plane.", flush=True)
    else:
        ground_size = max(6.0, 3.0 + 1.35 * len(loaded))
        add_ground(ground_size)

    print("[5/8] Positioning camera...", flush=True)
    camera = set_camera(model_objects)

    print("[6/8] Adding labels and title...", flush=True)
    add_family_labels([name for name, _, _ in loaded], model_objects, camera)
    add_3d_title(args.title, model_objects, camera)

    print("[7/8] Configuring renderer...", flush=True)
    configure_render(args, output_path)

    print("[8/8] Rendering...", flush=True)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered image: {output_path}", flush=True)


if __name__ == "__main__":
    main()
