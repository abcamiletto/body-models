"""Generate G1 test assets from the public Hugging Face MuJoCo asset repo."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from body_models.g1.io import G1_HF_BASE_URL, G1_HF_XML, G1_MESH_JOINT_MAP


def run(output_dir: Path) -> None:
    model_dir = output_dir / "g1" / "model"
    xml_dir = model_dir / "xml"
    mesh_dir = model_dir / "meshes" / "g1"
    xml_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(f"{G1_HF_BASE_URL}/{G1_HF_XML}", xml_dir / "g1.xml")
    for mesh_name in sorted({mesh for meshes in G1_MESH_JOINT_MAP.values() for mesh in meshes}):
        urllib.request.urlretrieve(f"{G1_HF_BASE_URL}/meshes/{mesh_name}", mesh_dir / mesh_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "assets",
        help="Directory where the g1/model asset tree will be written.",
    )
    run(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
