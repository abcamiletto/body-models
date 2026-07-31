"""Render lightweight front-view previews from the generated GLBs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    glbs = sorted(args.artifact_dir.glob("*.glb"))
    pngs = []
    for glb in glbs:
        png = glb.with_suffix(".png")
        _render(glb, png)
        pngs.append(png)
    _contact_sheet(pngs, args.artifact_dir / "contact_sheet.png")
    print(f"Wrote {len(pngs)} previews and contact_sheet.png")


def _render(glb: Path, output: Path) -> None:
    scene = trimesh.load_scene(glb)
    polygons = []
    colors = []
    depths = []
    light = np.array([-0.35, 0.45, 0.82])
    light /= np.linalg.norm(light)
    all_vertices = []

    for mesh in scene.dump():
        triangles = np.asarray(mesh.triangles)
        all_vertices.append(np.asarray(mesh.vertices))
        normal = np.asarray(mesh.vertex_normals)[np.asarray(mesh.faces)].mean(axis=1)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
        shade = np.clip(0.28 + 0.72 * np.abs(normal @ light), 0.0, 1.0)
        material = getattr(mesh.visual, "material", None)
        factor = getattr(material, "baseColorFactor", None)
        if factor is None:
            factor = [0.82, 0.88, 0.94, 1.0]
        base = np.asarray(factor, dtype=float)
        if base.max() > 1.0:
            base /= 255.0
        rgb = np.clip(base[:3][None] * shade[:, None], 0.0, 1.0)
        polygons.extend(triangles[:, :, :2])
        colors.extend(rgb)
        depths.extend(triangles[:, :, 2].mean(axis=1))

    order = np.argsort(depths)
    polygons = np.asarray(polygons)[order]
    colors = np.asarray(colors)[order]
    bounds = np.vstack(all_vertices)
    low = bounds.min(axis=0)
    high = bounds.max(axis=0)
    margin = 0.07 * max(high[0] - low[0], high[1] - low[1])

    figure, axis = plt.subplots(figsize=(4, 6), dpi=180)
    figure.patch.set_facecolor("#050a12")
    axis.set_facecolor("#050a12")
    axis.add_collection(PolyCollection(polygons, facecolors=colors, edgecolors="none"))
    axis.set_xlim(low[0] - margin, high[0] + margin)
    axis.set_ylim(low[1] - margin, high[1] + margin)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(glb.stem.replace("_", " ").upper(), color="#dce8f4", fontsize=10, pad=8)
    figure.savefig(output, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def _contact_sheet(pngs: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in pngs]
    if not images:
        return
    columns = min(4, len(images))
    rows = (len(images) + columns - 1) // columns
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (columns * width, rows * height), "#050a12")
    for index, image in enumerate(images):
        x = (index % columns) * width + (width - image.width) // 2
        y = (index // columns) * height + (height - image.height) // 2
        sheet.paste(image, (x, y))
    ImageDraw.Draw(sheet).rectangle((0, 0, sheet.width - 1, sheet.height - 1), outline="#26384a", width=2)
    sheet.save(output)


if __name__ == "__main__":
    main()
