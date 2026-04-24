"""Generate the minimal G1 fixture used by the test asset bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from body_models.g1.io import G1_MESH_JOINT_MAP, JOINT_NAMES, PARENTS


def _body_name(joint_name: str) -> str:
    if joint_name == "pelvis_skel":
        return "pelvis"
    if joint_name.endswith("_skel"):
        return f"{joint_name.removesuffix('_skel')}_link"
    return joint_name


def _joint_xml(joint_name: str) -> str:
    if joint_name == "pelvis_skel":
        return '<freejoint name="floating_base_joint"/>'
    if joint_name == "left_hip_pitch_skel":
        return '<joint name="left_hip_pitch_joint" axis="0 1 0" range="-0.5 0.5"/>'
    return ""


def _body_pos(joint_name: str) -> str:
    if joint_name == "left_hip_pitch_skel":
        return "1 0 0"
    return "0 0 0"


def _geom_pos(mesh_name: str) -> str:
    if mesh_name == "pelvis.STL":
        return ' pos="0 0 1"'
    if mesh_name == "left_hip_pitch_link.STL":
        return ' pos="0 1 0"'
    return ""


def _geom_xml(joint_name: str) -> list[str]:
    return [
        f'<geom class="visual" mesh="{mesh_name.removesuffix(".STL")}"{_geom_pos(mesh_name)}/>'
        for mesh_name in G1_MESH_JOINT_MAP.get(joint_name, [])
    ]


def _body_xml(joint_idx: int, children: dict[int, list[int]], indent: int = 2) -> list[str]:
    joint_name = JOINT_NAMES[joint_idx]
    pad = " " * indent
    lines = [f'{pad}<body name="{_body_name(joint_name)}" pos="{_body_pos(joint_name)}">']
    if joint_xml := _joint_xml(joint_name):
        lines.append(f"{pad}  {joint_xml}")
    lines.extend(f"{pad}  {geom_xml}" for geom_xml in _geom_xml(joint_name))
    for child_idx in children[joint_idx]:
        lines.extend(_body_xml(child_idx, children, indent + 2))
    lines.append(f"{pad}</body>")
    return lines


def _xml() -> str:
    children = {i: [] for i in range(len(JOINT_NAMES))}
    for joint_idx, parent_idx in enumerate(PARENTS):
        if parent_idx >= 0:
            children[parent_idx].append(joint_idx)

    mesh_names = sorted({mesh for meshes in G1_MESH_JOINT_MAP.values() for mesh in meshes})
    asset_lines = [f'    <mesh name="{mesh.removesuffix(".STL")}" file="{mesh}"/>' for mesh in mesh_names]
    body_lines = _body_xml(0, children, indent=4)
    return "\n".join(
        [
            '<mujoco model="g1_test">',
            '  <compiler angle="radian" meshdir="../meshes/g1"/>',
            "",
            "  <asset>",
            *asset_lines,
            "  </asset>",
            "",
            "  <worldbody>",
            *body_lines,
            "  </worldbody>",
            "</mujoco>",
            "",
        ]
    )

STL = """solid {name}
  facet normal 0 0 1
    outer loop
      vertex 1 0 0
      vertex 0 1 0
      vertex 0 0 1
    endloop
  endfacet
endsolid {name}
"""


def run(output_dir: Path) -> None:
    model_dir = output_dir / "g1" / "model"
    xml_dir = model_dir / "xml"
    mesh_dir = model_dir / "meshes" / "g1"
    xml_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    (xml_dir / "g1.xml").write_text(_xml())
    for mesh_name in {mesh for meshes in G1_MESH_JOINT_MAP.values() for mesh in meshes}:
        (mesh_dir / mesh_name).write_text(STL.format(name=mesh_name.removesuffix(".STL")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "assets",
        help="Directory where the g1/model fixture tree will be written.",
    )
    run(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
