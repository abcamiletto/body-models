"""Generate the minimal G1 fixture used by the test asset bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

XML = """<mujoco model="g1_test">
  <compiler angle="radian" meshdir="../meshes/g1"/>

  <default>
    <default class="g1">
      <default class="hip_pitch">
        <joint axis="0 1 0" range="-0.5 0.5"/>
      </default>
    </default>
  </default>

  <asset>
    <mesh name="pelvis" file="pelvis.STL"/>
    <mesh name="left_hip_pitch_link" file="left_hip_pitch_link.STL"/>
  </asset>

  <worldbody>
    <body name="pelvis" pos="0 0 0.793" childclass="g1">
      <freejoint name="floating_base_joint"/>
      <geom class="visual" mesh="pelvis" pos="0 0 1"/>
      <body name="left_hip_pitch_link" pos="1 0 0">
        <joint name="left_hip_pitch_joint" class="hip_pitch"/>
        <geom class="visual" mesh="left_hip_pitch_link" pos="0 1 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

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

    (xml_dir / "g1.xml").write_text(XML)
    for name in ["pelvis", "left_hip_pitch_link"]:
        (mesh_dir / f"{name}.STL").write_text(STL.format(name=name))


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
