"""Open the authored robot in Blender and start the Blender MCP socket."""

from __future__ import annotations

from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[2]
BLEND_PATH = ROOT / "src/body_models/robots/smpl_humanoid/assets/smpl_robot_professional.blend"
ADDON_PATH = Path("/Users/abcamiletto/.local/share/blender-mcp/blender_mcp_addon.py")


def main() -> None:
    if bpy.app.background:
        raise RuntimeError("Blender MCP requires a GUI Blender process.")
    if not ADDON_PATH.is_file():
        raise FileNotFoundError(ADDON_PATH)

    module_name = "blender_mcp_addon"
    if module_name not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_install(filepath=str(ADDON_PATH))
        bpy.ops.preferences.addon_enable(module=module_name)
        bpy.ops.wm.save_userpref()

    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    if module_name not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module=module_name)
    bpy.context.scene.blendermcp_auto_start_server = True
    bpy.context.scene.blendermcp_port = 9876
    bpy.ops.blendermcp.start_server()
    print("BLENDER_MCP_READY localhost:9876", flush=True)


if __name__ == "__main__":
    main()
