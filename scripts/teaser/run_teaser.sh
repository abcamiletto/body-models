#!/usr/bin/env bash
set -euo pipefail

# Regenerate the README teaser lineup image.
# Usage:
#   scripts/teaser/run_teaser.sh [output_png]
# Environment overrides:
#   ENGINE=CYCLES|BLENDER_EEVEE
#   SAMPLES=512
#   MAX_BOUNCES=14
#   DIFFUSE_BOUNCES=6
#   GLOSSY_BOUNCES=6
#   TRANSMISSION_BOUNCES=8
#   TRANSPARENT_MAX_BOUNCES=16
#   VOLUME_BOUNCES=2
#   ADAPTIVE_THRESHOLD=0.003
#   DENOISE=0|1
#   WIDTH=2200
#   HEIGHT=1200
#   MESH_DIR=/abs/or/rel/path

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_PATH="${1:-$ROOT_DIR/assets/readme/body-model-lineup.png}"
MESH_DIR="${MESH_DIR:-$ROOT_DIR/scripts/teaser/.meshes}"
ENGINE="${ENGINE:-CYCLES}"
SAMPLES="${SAMPLES:-512}"
MAX_BOUNCES="${MAX_BOUNCES:-14}"
DIFFUSE_BOUNCES="${DIFFUSE_BOUNCES:-6}"
GLOSSY_BOUNCES="${GLOSSY_BOUNCES:-6}"
TRANSMISSION_BOUNCES="${TRANSMISSION_BOUNCES:-8}"
TRANSPARENT_MAX_BOUNCES="${TRANSPARENT_MAX_BOUNCES:-16}"
VOLUME_BOUNCES="${VOLUME_BOUNCES:-2}"
ADAPTIVE_THRESHOLD="${ADAPTIVE_THRESHOLD:-0.003}"
DENOISE="${DENOISE:-0}"
WIDTH="${WIDTH:-2200}"
HEIGHT="${HEIGHT:-1200}"

mkdir -p "$(dirname "$OUTPUT_PATH")" "$MESH_DIR"
cd "$ROOT_DIR"

uv run python scripts/teaser/export_readme_meshes.py \
  --out "$MESH_DIR" \
  --families smpl smplx skel mhr anny

render_args=(
  --mesh-dir "$MESH_DIR"
  --skip-export
  --output "$OUTPUT_PATH"
  --engine "$ENGINE"
  --samples "$SAMPLES"
  --max-bounces "$MAX_BOUNCES"
  --diffuse-bounces "$DIFFUSE_BOUNCES"
  --glossy-bounces "$GLOSSY_BOUNCES"
  --transmission-bounces "$TRANSMISSION_BOUNCES"
  --transparent-max-bounces "$TRANSPARENT_MAX_BOUNCES"
  --volume-bounces "$VOLUME_BOUNCES"
  --adaptive-threshold "$ADAPTIVE_THRESHOLD"
  --width "$WIDTH"
  --height "$HEIGHT"
)

if [[ "$DENOISE" == "1" ]]; then
  render_args+=(--denoise)
fi

blender -b --python-exit-code 1 -P scripts/teaser/render_readme_blender.py -- "${render_args[@]}"

echo "Teaser image written to: $OUTPUT_PATH"
