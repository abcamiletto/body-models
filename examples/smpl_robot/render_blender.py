"""Render all rigid SMPL Robot GLBs and make a contact sheet."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def main() -> None:
    output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    glbs = sorted(output.glob("*.glb"))
    pngs = []
    for glb in glbs:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        bpy.ops.import_scene.gltf(filepath=str(glb))
        _setup_scene()
        png = output / f"{glb.stem}.png"
        bpy.context.scene.render.filepath = str(png)
        bpy.ops.render.render(write_still=True)
        pngs.append(png)
    if pngs:
        _contact_sheet(pngs, output / "contact_sheet.png")


def _contact_sheet(paths: list[Path], output: Path) -> None:
    images = [bpy.data.images.load(str(path), check_existing=False) for path in paths]
    width, height = images[0].size
    columns = min(4, len(images))
    rows = math.ceil(len(images) / columns)
    canvas = np.zeros((rows * height, columns * width, 4), dtype=np.float32)
    canvas[..., 0] = 0.003
    canvas[..., 1] = 0.006
    canvas[..., 2] = 0.012
    canvas[..., 3] = 1.0
    for index, image in enumerate(images):
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        pixels = pixels.reshape(height, width, 4)
        column = index % columns
        row = rows - 1 - index // columns
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = pixels
    sheet = bpy.data.images.new("Review contact sheet", width=columns * width, height=rows * height)
    sheet.pixels.foreach_set(canvas.reshape(-1))
    sheet.filepath_raw = str(output)
    sheet.file_format = "PNG"
    sheet.save()
    for image in images:
        bpy.data.images.remove(image)
    bpy.data.images.remove(sheet)


def _setup_scene() -> None:
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    low = Vector((math.inf, math.inf, math.inf))
    high = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        bevel = obj.modifiers.new("Machined edges", "BEVEL")
        bevel.width = 0.0015
        bevel.segments = 3
        bevel.limit_method = "ANGLE"
        bevel.angle_limit = math.radians(38)
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            low = Vector(map(min, low, point))
            high = Vector(map(max, high, point))
    center = (low + high) * 0.5
    height = high.z - low.z

    floor_data = bpy.data.meshes.new("Studio floor")
    floor_data.from_pydata(
        [
            (-3.0 * height, -3.0 * height, low.z - 0.012),
            (3.0 * height, -3.0 * height, low.z - 0.012),
            (3.0 * height, 3.0 * height, low.z - 0.012),
            (-3.0 * height, 3.0 * height, low.z - 0.012),
        ],
        [],
        [(0, 1, 2, 3)],
    )
    floor = bpy.data.objects.new("Studio floor", floor_data)
    bpy.context.collection.objects.link(floor)
    floor_material = bpy.data.materials.new("Studio floor")
    floor_material.diffuse_color = (0.34, 0.32, 0.29, 1.0)
    floor_material.metallic = 0.0
    floor_material.roughness = 0.72
    floor.data.materials.append(floor_material)

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + Vector((0.20 * height, -2.55 * height, 0.12 * height))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 64
    bpy.context.scene.camera = camera

    for name, energy, offset, size in (
        ("Key", 1100, (-1.4, -1.8, 1.8), 3.0),
        ("Fill", 700, (1.5, -1.0, 0.8), 4.0),
        ("Rim", 1200, (0.0, 1.6, 1.4), 3.0),
        ("Top", 800, (0.0, 0.0, 2.5), 2.5),
    ):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.size = size
        light = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(light)
        light.location = center + Vector(offset) * height
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 600
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.018, 0.016, 0.014, 1.0)
    background.inputs["Strength"].default_value = 0.35
    scene.view_settings.look = "AgX - Medium High Contrast"


if __name__ == "__main__":
    main()
