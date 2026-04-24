# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "bpy>=4.2.0",
#   "numpy>=1.26,<2",
# ]
# ///
"""Generate the private GarmentMeasurements rig asset from upstream data."""

from __future__ import annotations

import sys
import runpy
from pathlib import Path


if __name__ == "__main__":
    script = Path(__file__).resolve().parents[2] / "src" / "body_models" / "garment_measurements" / "generate_asset.py"
    main = runpy.run_path(str(script))["main"]
    main(sys.argv[1:])
