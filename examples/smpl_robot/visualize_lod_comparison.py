"""Compare rigid SMPL mannequin LODs under synchronized joint controls."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import viser
from nanomanifold import SO3
from visualize_viser import JOINT_GROUPS, add_rigid_meshes, read_link_colors

from body_models.smpl_humanoid import SmplMannequin

MODEL_SEPARATION = 0.95


def build_viewer(
    model_paths: list[Path],
    *,
    host: str,
    port: int,
) -> viser.ViserServer:
    if len(model_paths) < 2:
        raise ValueError("Pass at least two mannequin XML files.")

    robots = [SmplMannequin(model_path=path) for path in model_paths]
    reference = robots[0]
    if any(reference.joint_names != robot.joint_names for robot in robots[1:]):
        raise ValueError("The comparison mannequins must use the same joint hierarchy.")

    grounds = [min(mesh.bounds[0, 1] for mesh in robot.forward_meshes(**robot.get_tpose())) for robot in robots]
    labels = [path.parent.name.replace("_", " ").upper() for path in model_paths]
    vertices = [int(robot._weights.vertices.shape[0]) for robot in robots]
    positions = (np.arange(len(robots)) - (len(robots) - 1) / 2.0) * MODEL_SEPARATION

    server = viser.ViserServer(host=host, port=port, label="Mannequin LOD comparison")
    server.gui.add_markdown(
        "\n".join(
            f"**{label}:** {vertex_count:,} vertices  " for label, vertex_count in zip(labels, vertices, strict=True)
        )
        + "\n"
        "Both use identical rigid links, joint centers, and pose controls."
    )
    server.scene.set_up_direction("+y")
    frames = [
        server.scene.add_frame(
            f"/model_{index}",
            show_axes=False,
            position=(float(x), -ground, 0.0),
        )
        for index, (x, ground) in enumerate(zip(positions, grounds, strict=True))
    ]
    grid = server.scene.add_grid(
        "/ground",
        width=max(4.0, len(robots) * 1.4),
        height=4.0,
        plane="xz",
        cell_size=0.1,
        section_size=0.5,
        cell_color=(205, 195, 180),
        section_color=(150, 135, 115),
        plane_color=(242, 235, 220),
        plane_opacity=0.35,
    )
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)

    mesh_groups = [
        add_rigid_meshes(
            server,
            robot,
            read_link_colors(path, robot.link_names),
            root=f"/model_{index}",
        )
        for index, (path, robot) in enumerate(zip(model_paths, robots, strict=True))
    ]

    pose_controls = {}
    control_names = reference.actuated_joint_names[::3]
    limits = np.rad2deg(np.asarray(reference.actuated_joint_limits)).reshape(len(control_names), 3, 2)
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
                )

    with server.gui.add_folder("Display"):
        visibility_controls = [server.gui.add_checkbox(f"Show {label}", initial_value=True) for label in labels]
        show_grid = server.gui.add_checkbox("Show ground", initial_value=True)

    syncing_controls = False

    def update() -> None:
        degrees = np.stack([np.asarray(pose_controls[name].value) for name in control_names])
        pose = np.deg2rad(degrees).astype(np.float32).reshape(-1)
        transform_groups = [np.asarray(robot.forward_links(body_pose=pose)) for robot in robots]
        with server.atomic():
            for handles, transforms in zip(mesh_groups, transform_groups, strict=True):
                for handle, transform in zip(handles, transforms, strict=True):
                    handle.position = transform[:3, 3]
                    handle.wxyz = SO3.conversions.from_rotmat_to_quat(
                        transform[:3, :3],
                        convention="wxyz",
                        xp=np,
                    )

    def set_pose(pose: dict[str, np.ndarray]) -> None:
        nonlocal syncing_controls
        degrees = np.rad2deg(np.asarray(pose["body_pose"])).reshape(len(control_names), 3)
        syncing_controls = True
        with server.atomic():
            for name, value in zip(control_names, degrees, strict=True):
                pose_controls[name].value = value
        syncing_controls = False
        update()

    def update_from_gui(_event=None) -> None:
        if not syncing_controls:
            update()

    for control in pose_controls.values():
        control.on_update(update_from_gui)

    for frame, control in zip(frames, visibility_controls, strict=True):
        control.on_update(lambda _event, frame=frame, control=control: setattr(frame, "visible", control.value))

    @show_grid.on_update
    def _(_event) -> None:
        grid.visible = show_grid.value

    with server.gui.add_folder("Pose presets"):
        t_pose_button = server.gui.add_button("T-pose")
        a_pose_button = server.gui.add_button("A-pose")

    @t_pose_button.on_click
    def _(_event) -> None:
        set_pose(reference.get_tpose())

    @a_pose_button.on_click
    def _(_event) -> None:
        set_pose(reference.get_apose())

    set_pose(reference.get_tpose())
    server.initial_camera.position = (0.0, 0.35, 5.5 + 0.65 * len(robots))
    server.initial_camera.look_at = (0.0, 0.9, 0.0)
    server.initial_camera.up = (0.0, 1.0, 0.0)
    server.initial_camera.fov = np.deg2rad(35.0)
    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = build_viewer(
        [path.resolve() for path in args.models],
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
