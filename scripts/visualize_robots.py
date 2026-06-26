from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import viser

from body_models.base import RigidBodyModel
from body_models.registry import create_model
from body_models.robots.smpl_humanoid import SMPL_HUMANOID_VARIANTS


def model_label(name: str) -> str:
    label = name.replace("_", " ").title()
    return label.replace("Smplsim", "SMPLSim").replace("Smpl", "SMPL").replace("Phc", "PHC")


SMPL_HUMANOID = model_label("humenv")
MODEL_SPECS: dict[str, tuple[str, dict[str, Any]]] = {
    "G1": ("g1", {}),
    "BrainCo Right": ("brainco", {"side": "right"}),
    "BrainCo Left": ("brainco", {"side": "left"}),
    "MyoFullBody": ("myofullbody", {}),
    **{model_label(name): (name, {}) for name in SMPL_HUMANOID_VARIANTS},
}
SMPL_HUMANOID_COLOR = (190, 190, 205)
MODEL_COLORS: dict[str, tuple[int, int, int]] = {
    "G1": (152, 190, 255),
    "BrainCo Right": (238, 180, 120),
    "BrainCo Left": (238, 180, 120),
    "MyoFullBody": (175, 210, 165),
}
GRID_COLS = 2
GRID_SPACING_X = 1.6
GRID_SPACING_Z = 1.5
DEFAULT_LIMITS = (-1.5, 1.5)


@dataclass
class RobotState:
    model: RigidBodyModel
    params: dict[str, np.ndarray]
    mesh_path: str
    color: tuple[int, int, int]
    mesh_handle: viser.MeshHandle | None = None
    changed_keys: set[str] = field(default_factory=set)
    hands: Literal["default", "flat", "rest"] = "default"


@dataclass
class SliderHandle:
    handle: viser.GuiInputHandle
    initial: float
    key: str
    indices: tuple[int, ...]


@dataclass
class RobotControls:
    folder: viser.GuiFolderHandle
    sliders: list[SliderHandle]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize rigid robot models.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("_VISER_PORT_OVERRIDE", "8080")))
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(MODEL_SPECS),
        help="Robot model to load.",
    )
    parser.add_argument("--smpl-humanoid-model-path", help="MJCF XML path for SmplHumanoid.")
    args = parser.parse_args()
    model_specs = specs(args.smpl_humanoid_model_path)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/grid", position=(0.0, 0.0, 0.0), plane="xz")
    server.gui.configure_theme(control_layout="fixed", control_width="large")

    models = load_models(model_specs, args.model)
    states = init_states(server, models)
    tabs = server.gui.add_tab_group()
    selected_model = next(iter(states))

    with tabs.add_tab("Robots", viser.Icon.ROBOT):
        model_dropdown = server.gui.add_dropdown("Robot", options=tuple(states), initial_value=selected_model)
        controls = {name: add_robot_controls(server, name, state) for name, state in states.items()}

    with tabs.add_tab("Presets"):
        with server.gui.add_folder("Hands"):
            for label, hands in (("Default hands", "default"), ("Flat hands", "flat"), ("Rest hands", "rest")):
                button = server.gui.add_button(label)

                @button.on_click
                def _(_, hands=hands) -> None:
                    for name, state in states.items():
                        if state.model.has_hands:
                            apply_hands(state, controls[name].sliders, hands)

    def show_robot_controls(name: str) -> None:
        for folder_name, robot_controls in controls.items():
            set_gui_visible(robot_controls.folder, folder_name == name)

    show_robot_controls(selected_model)

    @model_dropdown.on_update
    def _(event) -> None:
        show_robot_controls(event.target.value)

    print(f"Loaded {len(states)} robots: {list(states)}", flush=True)
    while True:
        time.sleep(0.02)
        for state in states.values():
            if state.changed_keys:
                state.changed_keys.clear()
                update_robot_mesh(server, state)


def specs(smpl_humanoid_model_path: str | None) -> dict[str, tuple[str, dict[str, Any]]]:
    model_specs = {name: (model_id, dict(kwargs)) for name, (model_id, kwargs) in MODEL_SPECS.items()}
    if smpl_humanoid_model_path is not None:
        model_specs[SMPL_HUMANOID] = ("smpl_humanoid", {"source": smpl_humanoid_model_path})
    return model_specs


def load_models(
    model_specs: dict[str, tuple[str, dict[str, Any]]], names: list[str] | None
) -> dict[str, RigidBodyModel]:
    models = {}
    for name in names or list(model_specs):
        print(f"Loading {name}", flush=True)
        model_id, kwargs = model_specs[name]
        model = create_model(model_id, backend="numpy", **kwargs)
        if not isinstance(model, RigidBodyModel):
            raise TypeError(f"{name} is not a rigid body model")
        models[name] = model
    return models


def init_states(server: viser.ViserServer, models: dict[str, RigidBodyModel]) -> dict[str, RobotState]:
    n = len(models)
    num_rows = (n + GRID_COLS - 1) // GRID_COLS
    states = {}
    for i, (name, model) in enumerate(models.items()):
        row, col = divmod(i, GRID_COLS)
        row_count = min(GRID_COLS, n - row * GRID_COLS)
        params = mutable_params(model.get_rest_pose())
        mesh_path = f"/robots/{name}"
        state = RobotState(
            model=model, params=params, mesh_path=mesh_path, color=MODEL_COLORS.get(name, SMPL_HUMANOID_COLOR)
        )
        update_robot_mesh(server, state)
        assert state.mesh_handle is not None
        bounds = np.asarray(state.mesh_handle.vertices)
        params["global_translation"] = np.asarray(
            (
                (col - 0.5 * (row_count - 1)) * GRID_SPACING_X,
                -float(bounds[..., 1].min()),
                (row - 0.5 * (num_rows - 1)) * GRID_SPACING_Z,
            ),
            dtype=params["global_translation"].dtype,
        )
        update_robot_mesh(server, state)
        add_label(server, name, state)
        states[name] = state
    return states


def mutable_params(params: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in params.items()}


def update_robot_mesh(server: viser.ViserServer, state: RobotState) -> None:
    mesh = state.model.forward_meshes(**state.params)[0]
    if state.mesh_handle is not None:
        state.mesh_handle.remove()
    state.mesh_handle = server.scene.add_mesh_simple(
        state.mesh_path,
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        color=state.color,
        side="double",
    )


def add_label(server: viser.ViserServer, name: str, state: RobotState) -> None:
    assert state.mesh_handle is not None
    vertices = np.asarray(state.mesh_handle.vertices)
    position = vertices.mean(axis=0)
    position[1] = float(vertices[:, 1].max()) + 0.1
    server.scene.add_label(f"/labels/{name}", text=name, position=position)


def pose_key(params: dict[str, np.ndarray]) -> str:
    return "hand_pose" if "hand_pose" in params else "body_pose"


def add_robot_controls(server: viser.ViserServer, name: str, state: RobotState) -> RobotControls:
    handles: list[SliderHandle] = []
    key = pose_key(state.params)
    with server.gui.add_folder(name, expand_by_default=True) as folder:
        with server.gui.add_folder("Pose"):
            for coord_index in range(state.model.num_actuated):
                lo, hi = slider_limits(state.model.actuated_joint_limits[coord_index])
                initial = float(state.params[key][coord_index])
                label = pose_slider_label(state.model.actuated_joint_names, coord_index)
                handles.append(
                    add_slider(
                        server,
                        state,
                        label,
                        lo=lo,
                        hi=hi,
                        step=0.01,
                        initial=initial,
                        key=key,
                        indices=(coord_index,),
                    )
                )
        reset_button(server, handles)
    return RobotControls(folder, handles)


def pose_slider_label(names: list[str], index: int) -> str:
    name = names[index]
    start = index
    while start > 0 and names[start - 1] == name:
        start -= 1
    stop = index + 1
    while stop < len(names) and names[stop] == name:
        stop += 1
    if stop - start == 3:
        return f"{name}_{'xyz'[index - start]}"
    if stop - start > 1:
        return f"{name}_{index - start}"
    return name


def slider_limits(limits: np.ndarray) -> tuple[float, float]:
    lo, hi = float(limits[0]), float(limits[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return DEFAULT_LIMITS
    return lo, hi


def add_slider(
    server: viser.ViserServer,
    state: RobotState,
    label: str,
    *,
    lo: float,
    hi: float,
    step: float,
    initial: float,
    key: str,
    indices: tuple[int, ...],
) -> SliderHandle:
    handle = server.gui.add_slider(label, min=lo, max=hi, step=step, initial_value=initial)

    @handle.on_update
    def _(event) -> None:
        state.params[key][indices] = event.target.value
        state.changed_keys.add(key)

    return SliderHandle(handle, initial, key, indices)


def reset_button(server: viser.ViserServer, handles: list[SliderHandle]) -> None:
    button = server.gui.add_button("Reset")

    @button.on_click
    def _(_) -> None:
        for slider in handles:
            slider.handle.value = slider.initial


def apply_hands(state: RobotState, sliders: list[SliderHandle], hands: Literal["default", "flat", "rest"]) -> None:
    preset = state.model.get_rest_pose(hands=hands)
    key = pose_key(state.params)
    state.hands = hands
    state.params[key] = np.asarray(preset[key]).copy()
    for slider in sliders:
        if slider.key == key:
            slider.handle.value = float(state.params[key][slider.indices])
    state.changed_keys.add(key)


def set_gui_visible(handle: Any, visible: bool) -> None:
    handle.visible = visible
    if isinstance(handle, viser.GuiFolderHandle):
        for child in handle._children.values():
            set_gui_visible(child, visible)


if __name__ == "__main__":
    main()
