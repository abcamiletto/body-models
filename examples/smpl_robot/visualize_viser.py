"""Compare the rigid SMPL mannequin and real SMPL-X under one interactive pose."""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import viser
from nanomanifold import SO3

from body_models.smpl_humanoid import SmplMannequin
from body_models.smpl_humanoid._constants import BODY_JOINTS, FINGER_CHAINS
from body_models.smplx import SMPLX

DEFAULT_MODEL = Path(__file__).parents[2] / "artifacts" / "smpl_robot" / "neutral.xml"
MODEL_SEPARATION = 1.0
BODY_INDEX_BY_NAME = dict(BODY_JOINTS)
SMPLX_BODY_CONTROLS = tuple((name, index) for name, index in BODY_JOINTS if index < SMPLX.NUM_BODY_JOINTS)
FINGER_NAMES = tuple(name for chain in FINGER_CHAINS for name in chain)
JOINT_GROUPS = {
    "Spine & head": ("Torso", "Spine", "Chest", "Neck", "Head"),
    "Left arm": ("L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"),
    "Right arm": ("R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"),
    "Left leg": ("L_Hip", "L_Knee", "L_Ankle", "L_Toe"),
    "Right leg": ("R_Hip", "R_Knee", "R_Ankle", "R_Toe"),
    "Left fingers": tuple(name for chain in FINGER_CHAINS[:5] for name in chain),
    "Right fingers": tuple(name for chain in FINGER_CHAINS[5:] for name in chain),
}


def smplx_pose_from_euler(
    control_names: list[str],
    euler_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map the rigid SMPL controls to equivalent SMPL-X body and hand poses."""
    euler_by_name = dict(zip(control_names, euler_pose, strict=True))
    body_pose = np.zeros((SMPLX.NUM_BODY_JOINTS, 3), dtype=np.float32)
    for name, smpl_index in SMPLX_BODY_CONTROLS:
        body_pose[smpl_index] = SO3.conversions.from_euler_to_axis_angle(
            euler_by_name[name],
            convention="XYZ",
            xp=np,
        )

    for side in ("L", "R"):
        wrist = SO3.conversions.from_euler_to_rotmat(
            euler_by_name[f"{side}_Wrist"],
            convention="XYZ",
            xp=np,
        )
        hand = SO3.conversions.from_euler_to_rotmat(
            euler_by_name[f"{side}_Hand"],
            convention="XYZ",
            xp=np,
        )
        wrist_index = BODY_INDEX_BY_NAME[f"{side}_Wrist"]
        body_pose[wrist_index] = SO3.conversions.from_rotmat_to_axis_angle(wrist @ hand, xp=np)

    hand_euler = np.stack([euler_by_name[name] for name in FINGER_NAMES])
    hand_pose = SO3.conversions.from_euler_to_axis_angle(
        hand_euler,
        convention="XYZ",
        xp=np,
    )
    return body_pose.astype(np.float32), hand_pose.astype(np.float32)


def read_link_colors(xml_path: Path, link_names: list[str]) -> list[tuple[int, int, int]]:
    """Read per-geometry RGB colors from the generated MJCF."""
    geoms = {geom.attrib["name"]: geom for geom in ET.parse(xml_path).getroot().findall(".//worldbody//geom")}
    colors = []
    for name in link_names:
        rgba = np.fromstring(geoms[name].attrib["rgba"], sep=" ")
        colors.append(tuple(np.clip(np.rint(rgba[:3] * 255.0), 0, 255).astype(int)))
    return colors


def add_rigid_meshes(
    server: viser.ViserServer,
    robot: SmplMannequin,
    colors: list[tuple[int, int, int]],
    *,
    root: str = "/mannequin",
) -> list[viser.MeshHandle]:
    """Add every geometry as an independent local-space mesh."""
    handles = []
    for link_index, (name, color) in enumerate(zip(robot.link_names, colors, strict=True)):
        vertex_start = robot.link_vertex_starts[link_index]
        vertex_count = robot.link_vertex_counts[link_index]
        face_start = robot.link_face_starts[link_index]
        face_count = robot.link_face_counts[link_index]
        vertices = robot._weights.vertices[vertex_start : vertex_start + vertex_count]
        faces = robot.faces[face_start : face_start + face_count] - vertex_start
        handles.append(
            server.scene.add_mesh_simple(
                f"{root}/rigid_links/{link_index:02d}_{name}",
                vertices=vertices,
                faces=faces,
                color=color,
                material="standard",
                flat_shading=False,
                side="double",
                cast_shadow=True,
                receive_shadow=True,
            )
        )
    return handles


def build_viewer(
    xml_path: Path,
    *,
    smplx_model: Path | None,
    host: str,
    port: int,
) -> viser.ViserServer:
    """Create the Viser scene and all interactive controls."""
    robot = SmplMannequin(model_path=xml_path)
    smplx = SMPLX(model_path=smplx_model, gender=None if smplx_model else "neutral", flat_hand_mean=True)
    smplx_identity = smplx.prepare_identity(
        np.zeros(10, dtype=np.float32),
        expression=np.zeros(10, dtype=np.float32),
    )
    smplx_vertices = np.asarray(smplx_identity["rest_vertices"])
    robot_ground_y = min(mesh.bounds[0, 1] for mesh in robot.forward_meshes(**robot.get_tpose()))
    smplx_ground_offset = robot_ground_y - float(smplx_vertices[:, 1].min())

    server = viser.ViserServer(host=host, port=port, label="Rigid mannequin + SMPL-X")
    server.gui.add_markdown("**Left:** rigid mannequin · **Right:** real SMPL-X")
    server.scene.set_up_direction("+y")
    mannequin_frame = server.scene.add_frame(
        "/mannequin",
        show_axes=False,
        position=(-MODEL_SEPARATION, 0.0, 0.0),
    )
    smplx_frame = server.scene.add_frame(
        "/smplx",
        show_axes=False,
        position=(MODEL_SEPARATION, smplx_ground_offset, 0.0),
    )
    grid_handle = server.scene.add_grid(
        "/ground",
        width=4.0,
        height=4.0,
        plane="xz",
        position=(0.0, robot_ground_y, 0.0),
        cell_size=0.1,
        section_size=0.5,
        cell_color=(205, 195, 180),
        section_color=(150, 135, 115),
        plane_color=(242, 235, 220),
        plane_opacity=0.35,
    )
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)

    mesh_handles = add_rigid_meshes(server, robot, read_link_colors(xml_path, robot.link_names))
    smplx_mesh = server.scene.add_mesh_simple(
        "/smplx/skinned_mesh",
        vertices=smplx_vertices,
        faces=smplx.faces,
        color=(164, 197, 207),
        material="standard",
        flat_shading=False,
        side="double",
        cast_shadow=True,
        receive_shadow=True,
    )
    joint_handles = [
        server.scene.add_icosphere(
            f"/mannequin/joints/{index:02d}_{name}",
            radius=0.014,
            color=(105, 172, 181),
            subdivisions=2,
        )
        for index, name in enumerate(robot.joint_names)
    ]
    smplx_joint_handles = [
        server.scene.add_icosphere(
            f"/smplx/joints/{index:02d}_{name}",
            radius=0.012,
            color=(214, 132, 113),
            subdivisions=2,
        )
        for index, name in enumerate(smplx.joint_names)
    ]

    pose_controls = {}
    with server.gui.add_folder("Root"):
        root_translation = server.gui.add_vector3(
            "Translation",
            initial_value=(0.0, 0.0, 0.0),
            step=0.01,
            hint="Global XYZ translation in meters.",
        )
        root_rotation = server.gui.add_vector3(
            "Rotation (axis-angle °)",
            initial_value=(0.0, 0.0, 0.0),
            min=(-180.0, -180.0, -180.0),
            max=(180.0, 180.0, 180.0),
            step=1.0,
        )

    control_names = robot.actuated_joint_names[::3]
    limits = np.rad2deg(np.asarray(robot.actuated_joint_limits)).reshape(len(control_names), 3, 2)
    limits_by_name = dict(zip(control_names, limits, strict=True))
    for group_name, joint_names in JOINT_GROUPS.items():
        with server.gui.add_folder(group_name, expand_by_default=group_name == "Spine & head"):
            for name in joint_names:
                limit = limits_by_name[name]
                pose_controls[name] = server.gui.add_vector3(
                    f"{name} XYZ (°)",
                    initial_value=(0.0, 0.0, 0.0),
                    min=limit[:, 0],
                    max=limit[:, 1],
                    step=1.0,
                    hint="Local SMPL joint rotation, applied in XYZ order.",
                )

    with server.gui.add_folder("Display"):
        show_mannequin = server.gui.add_checkbox("Show rigid mannequin", initial_value=True)
        show_smplx = server.gui.add_checkbox("Show real SMPL-X", initial_value=True)
        show_joints = server.gui.add_checkbox("Show joint centers", initial_value=False)
        show_grid = server.gui.add_checkbox("Show ground", initial_value=True)
    for handle in [*joint_handles, *smplx_joint_handles]:
        handle.visible = False

    syncing_controls = False
    head_pose = np.zeros((SMPLX.NUM_HEAD_JOINTS, 3), dtype=np.float32)

    def update() -> None:
        degrees = np.stack([np.asarray(pose_controls[name].value) for name in control_names])
        euler_pose = np.deg2rad(degrees).astype(np.float32)
        body_pose, hand_pose = smplx_pose_from_euler(control_names, euler_pose)
        link_transforms = np.asarray(robot.forward_links(body_pose=euler_pose.reshape(-1)))
        robot_skeleton = np.asarray(robot.forward_skeleton(body_pose=euler_pose.reshape(-1)))
        smplx_params = {
            "body_pose": body_pose,
            "hand_pose": hand_pose,
            "head_pose": head_pose,
            "identity": smplx_identity,
        }
        vertices = np.asarray(smplx.forward_vertices(**smplx_params))
        smplx_skeleton = np.asarray(smplx.forward_skeleton(**smplx_params))
        translation = np.asarray(root_translation.value, dtype=np.float32)
        rotation = np.deg2rad(np.asarray(root_rotation.value, dtype=np.float32))
        quaternion = SO3.conversions.from_axis_angle_to_quat(rotation, convention="wxyz", xp=np)

        with server.atomic():
            mannequin_frame.position = translation + (-MODEL_SEPARATION, 0.0, 0.0)
            smplx_frame.position = translation + (MODEL_SEPARATION, smplx_ground_offset, 0.0)
            mannequin_frame.wxyz = quaternion
            smplx_frame.wxyz = quaternion
            for handle, transform in zip(mesh_handles, link_transforms, strict=True):
                handle.position = transform[:3, 3]
                handle.wxyz = SO3.conversions.from_rotmat_to_quat(
                    transform[:3, :3],
                    convention="wxyz",
                    xp=np,
                )
            for handle, transform in zip(joint_handles, robot_skeleton, strict=True):
                handle.position = transform[:3, 3]
            smplx_mesh.vertices = vertices
            for handle, transform in zip(smplx_joint_handles, smplx_skeleton, strict=True):
                handle.position = transform[:3, 3]

    def set_pose(pose: dict[str, np.ndarray]) -> None:
        nonlocal syncing_controls
        degrees = np.rad2deg(np.asarray(pose["body_pose"])).reshape(len(control_names), 3)
        syncing_controls = True
        with server.atomic():
            for name, value in zip(control_names, degrees, strict=True):
                pose_controls[name].value = value
            root_translation.value = np.asarray(pose["global_translation"])
            root_rotation.value = np.rad2deg(np.asarray(pose["global_rotation"]))
        syncing_controls = False
        update()

    def update_from_gui(_event=None) -> None:
        if not syncing_controls:
            update()

    for control in [*pose_controls.values(), root_translation, root_rotation]:
        control.on_update(update_from_gui)

    @show_mannequin.on_update
    def _(_event) -> None:
        mannequin_frame.visible = show_mannequin.value

    @show_smplx.on_update
    def _(_event) -> None:
        smplx_frame.visible = show_smplx.value

    @show_joints.on_update
    def _(_event) -> None:
        for handle in [*joint_handles, *smplx_joint_handles]:
            handle.visible = show_joints.value

    @show_grid.on_update
    def _(_event) -> None:
        grid_handle.visible = show_grid.value

    with server.gui.add_folder("Pose presets"):
        t_pose_button = server.gui.add_button("T-pose")
        a_pose_button = server.gui.add_button("A-pose")

    @t_pose_button.on_click
    def _(_event) -> None:
        set_pose(robot.get_tpose())

    @a_pose_button.on_click
    def _(_event) -> None:
        set_pose(robot.get_apose())

    set_pose(robot.get_tpose())
    server.initial_camera.position = (0.0, 0.3, 7.0)
    server.initial_camera.look_at = (0.0, -0.15, 0.0)
    server.initial_camera.up = (0.0, 1.0, 0.0)
    server.initial_camera.fov = np.deg2rad(35.0)
    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL, help="Generated rigid mannequin MJCF.")
    parser.add_argument(
        "--smplx-model",
        type=Path,
        help="Neutral SMPL-X .npz/.pkl. Defaults to the configured smplx-neutral model.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_path = args.model.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"Generated mannequin not found: {xml_path}")
    server = build_viewer(
        xml_path,
        smplx_model=args.smplx_model,
        host=args.host,
        port=args.port,
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
