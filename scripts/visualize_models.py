from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import body_models_viser as bmv
import numpy as np
import viser

from body_models.anny.numpy import ANNY
from body_models.base import BodyModel
from body_models.flame.numpy import FLAME
from body_models.garment_measurements.numpy import GarmentMeasurements
from body_models.mano.numpy import MANO
from body_models.mhr.numpy import MHR
from body_models.skel.numpy import SKEL
from body_models.smpl.numpy import SMPL
from body_models.smplh.numpy import SMPLH
from body_models.smplx.numpy import SMPLX
from body_models.soma.numpy import SOMA


DISPLAY_GLOBAL_ROTATIONS = {
    "ANNY": (-np.pi / 2, 0.0, 0.0),
}

SMPL_POSE_JOINTS = [
    ("Spine1", 2),
    ("Spine2", 5),
    ("Spine3", 8),
    ("Neck", 11),
    ("L Shoulder", 15),
    ("R Shoulder", 16),
]
SKEL_POSE_DOFS = [
    ("Hip R flex", 3, (-1.5, 1.5)),
    ("Hip R abd", 4, (-0.8, 0.8)),
    ("Hip R rot", 5, (-0.8, 0.8)),
    ("Knee R", 6, (-2.5, 0.1)),
    ("Hip L flex", 7, (-1.5, 1.5)),
    ("Hip L abd", 8, (-0.8, 0.8)),
    ("Hip L rot", 9, (-0.8, 0.8)),
    ("Knee L", 10, (-2.5, 0.1)),
    ("Lumbar flex", 17, (-0.8, 0.8)),
    ("Lumbar bend", 18, (-0.5, 0.5)),
    ("Thorax flex", 20, (-0.5, 0.5)),
    ("Shoulder R", 28, (-1.5, 1.5)),
    ("Shoulder L", 38, (-1.5, 1.5)),
]
FLAME_POSE_JOINTS = [("Neck", 0), ("Jaw", 1), ("L Eye", 2), ("R Eye", 3)]
ANNY_PHENOTYPE_PARAMS = ["Gender", "Age", "Muscle", "Weight", "Height", "Proportions"]
ANNY_BODY_POSE_BONES = [
    ("Spine", 0),
    ("Spine1", 1),
    ("Spine2", 2),
    ("Neck", 3),
    ("L Shoulder", 7),
    ("L Arm", 8),
    ("R Shoulder", 12),
    ("R Arm", 13),
    ("L UpLeg", 17),
    ("R UpLeg", 22),
]
GARMENT_MEASUREMENTS_BODY_POSE_JOINTS = [
    ("Spine1", 0),
    ("Spine2", 1),
    ("Chest", 2),
    ("Neck1", 3),
    ("L Clavicle", 5),
    ("L UpperArm", 6),
    ("R Clavicle", 11),
    ("R UpperArm", 12),
    ("L Thigh", 17),
    ("R Thigh", 21),
]
GARMENT_MEASUREMENTS_HEAD_POSE_JOINTS = [("Head", 0)]
SOMA_BODY_POSE_JOINTS = [
    ("Spine1", 0),
    ("Spine2", 1),
    ("Chest", 2),
    ("Neck1", 3),
    ("L Shoulder", 5),
    ("L Arm", 6),
    ("L ForeArm", 7),
    ("R Shoulder", 9),
    ("R Arm", 10),
    ("R ForeArm", 11),
    ("L Leg", 13),
    ("R Leg", 18),
]
SOMA_HEAD_POSE_JOINTS = [("Head", 0)]

GRID_COLS = 5
GRID_SPACING_X = 1.8
GRID_SPACING_Z = 1.8

MODEL_COLORS: dict[str, tuple[int, int, int]] = {
    "SMPL": (173, 216, 230),
    "SMPLH": (216, 191, 216),
    "MANO": (245, 205, 155),
    "SMPLX": (255, 182, 193),
    "SKEL": (144, 238, 144),
    "ANNY": (255, 218, 185),
    "MHR": (221, 160, 221),
    "FLAME": (255, 239, 186),
    "GarmentMeasurements": (176, 224, 230),
    "SOMA": (250, 200, 200),
}
CANONICAL_POSE_MODELS = ("SMPL", "SMPLH", "SMPLX", "SKEL", "ANNY", "MHR", "GarmentMeasurements", "SOMA")


@dataclass
class ModelState:
    model: BodyModel
    params: dict[str, np.ndarray]
    color: tuple[int, int, int]
    display_global_rotation: np.ndarray | None = None
    hands: str = "default"
    body_handle: bmv.BodyModelHandle | None = None
    changed: bool = True


@dataclass
class SliderHandle:
    handle: viser.GuiInputHandle
    initial: float
    key: str
    indices: tuple[int, ...]


@dataclass
class ModelControls:
    folder: viser.GuiFolderHandle
    sliders: list[SliderHandle]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize body models with body-models-viser.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("_VISER_PORT_OVERRIDE", "8080")))
    parser.add_argument("--model", action="append", choices=sorted(model_specs()), help="Model to load.")
    args = parser.parse_args()

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/grid", position=(0.0, 0.0, 0.0), plane="xz")
    server.gui.configure_theme(control_layout="fixed", control_width="large")

    models = load_models(args.model)
    states = init_states(models)
    tabs = server.gui.add_tab_group()
    selected_model = next(iter(states))

    with tabs.add_tab("Models", viser.Icon.USER):
        model_dropdown = server.gui.add_dropdown("Model", options=tuple(states), initial_value=selected_model)
        controls = {name: add_model_controls(server, name, state) for name, state in states.items()}

    with tabs.add_tab("Poses"):
        with server.gui.add_folder("Body"):
            for label, pose_name in (("T-pose", "tpose"), ("A-pose", "apose")):
                button = server.gui.add_button(label)

                @button.on_click
                def _(_, pose_name=pose_name) -> None:
                    for name in CANONICAL_POSE_MODELS:
                        if name in states:
                            apply_pose(states[name], controls[name].sliders, pose_name)

        with server.gui.add_folder("Hands"):
            for label, hands in (("Default hands", "default"), ("Flat hands", "flat"), ("Rest hands", "rest")):
                button = server.gui.add_button(label)

                @button.on_click
                def _(_, hands=hands) -> None:
                    for name, state in states.items():
                        if state.model.has_hands:
                            apply_hands(state, controls[name].sliders, hands)

    def show_model_controls(name: str) -> None:
        for folder_name, model_controls in controls.items():
            set_gui_visible(model_controls.folder, folder_name == name)

    show_model_controls(selected_model)

    @model_dropdown.on_update
    def _(event) -> None:
        show_model_controls(event.target.value)

    add_labels(server, states)
    print(f"Loaded {len(states)} models: {list(states)}", flush=True)

    while True:
        time.sleep(0.02)
        for name, state in states.items():
            if state.changed:
                state.changed = False
                update_body_handle(server, name, state)


def model_specs() -> dict[str, Callable[[], BodyModel]]:
    return {
        "SMPL": lambda: SMPL(gender="neutral"),
        "SMPLH": lambda: SMPLH(gender="neutral"),
        "MANO": lambda: MANO(side="right"),
        "SMPLX": lambda: SMPLX(gender="neutral"),
        "SKEL": lambda: SKEL(gender="male"),
        "ANNY": ANNY,
        "MHR": MHR,
        "FLAME": FLAME,
        "GarmentMeasurements": GarmentMeasurements,
        "SOMA": SOMA,
    }


def load_models(names: list[str] | None) -> dict[str, BodyModel]:
    specs = model_specs()
    models = {}
    for name in names or list(specs):
        try:
            print(f"Loading {name}", flush=True)
            models[name] = specs[name]()
        except Exception as exc:
            print(f"Skipping {name}: {exc}", flush=True)
    if not models:
        raise RuntimeError("No models loaded.")
    return models


def init_states(models: dict[str, BodyModel]) -> dict[str, ModelState]:
    n = len(models)
    num_rows = (n + GRID_COLS - 1) // GRID_COLS
    states = {}
    for i, (name, model) in enumerate(models.items()):
        row, col = divmod(i, GRID_COLS)
        row_count = min(GRID_COLS, n - row * GRID_COLS)
        params = mutable_params(model.get_rest_pose())
        display_global_rotation = DISPLAY_GLOBAL_ROTATIONS.get(name)
        if display_global_rotation is not None:
            display_global_rotation = np.asarray(display_global_rotation, dtype=params["global_rotation"].dtype)
            params["global_rotation"] = display_global_rotation.copy()
        verts = model.forward_vertices(**params)
        params["global_translation"] = np.asarray(
            (
                (col - 0.5 * (row_count - 1)) * GRID_SPACING_X,
                -float(verts[..., 1].min()),
                (row - 0.5 * (num_rows - 1)) * GRID_SPACING_Z,
            ),
            dtype=params["global_translation"].dtype,
        )
        states[name] = ModelState(
            model=model,
            params=params,
            color=MODEL_COLORS[name],
            display_global_rotation=display_global_rotation,
        )
    return states


def mutable_params(params: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in params.items()}


def update_body_handle(server: viser.ViserServer, name: str, state: ModelState) -> None:
    if state.body_handle is None:
        state.body_handle = bmv.add_body_model(server.scene, f"/meshes/{name}", state.model, color=state.color)
    state.body_handle.set_pose(**state.params)


def add_labels(server: viser.ViserServer, states: dict[str, ModelState]) -> None:
    for name, state in states.items():
        verts = state.model.forward_vertices(**state.params)
        label_position = np.asarray(state.params["global_translation"]).copy()
        label_position[1] = float(verts[..., 1].max()) + 0.1
        server.scene.add_label(f"/labels/{name}", text=name, position=label_position)


def add_slider(
    server: viser.ViserServer,
    state: ModelState,
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
        state.changed = True

    return SliderHandle(handle, initial, key, indices)


def betas(
    server: viser.ViserServer,
    state: ModelState,
    *,
    key: str,
    count: int,
    prefix: str = "beta",
    lo: float = -3.0,
    hi: float = 3.0,
    step: float = 0.1,
    initial: float = 0.0,
) -> list[SliderHandle]:
    return [
        add_slider(
            server,
            state,
            f"{prefix}{i}",
            lo=lo,
            hi=hi,
            step=step,
            initial=initial,
            key=key,
            indices=(i,),
        )
        for i in range(count)
    ]


def joint_xyz(
    server: viser.ViserServer,
    state: ModelState,
    *,
    key: str,
    joints: list[tuple[str, int]],
    lo: float = -1.5,
    hi: float = 1.5,
    step: float = 0.05,
    max_joints: int | None = None,
) -> list[SliderHandle]:
    handles: list[SliderHandle] = []
    for name, idx in joints:
        if max_joints is not None and idx >= max_joints:
            continue
        for axis, axis_name in enumerate("XYZ"):
            handles.append(
                add_slider(
                    server,
                    state,
                    f"{name} {axis_name}",
                    lo=lo,
                    hi=hi,
                    step=step,
                    initial=0.0,
                    key=key,
                    indices=(idx, axis),
                )
            )
    return handles


def add_model_controls(server: viser.ViserServer, name: str, state: ModelState) -> ModelControls:
    handles: list[SliderHandle] = []
    with server.gui.add_folder(name, expand_by_default=True) as folder:
        if name in {"SMPL", "SMPLH", "SMPLX"}:
            with server.gui.add_folder("Shape"):
                handles += betas(server, state, key="shape", count=10)
            if name == "SMPLX":
                with server.gui.add_folder("Expression"):
                    handles += betas(server, state, key="expression", count=10, prefix="psi", lo=-2.0, hi=2.0)
            with server.gui.add_folder("Body Pose"):
                handles += joint_xyz(server, state, key="body_pose", joints=SMPL_POSE_JOINTS)
        elif name == "MANO":
            with server.gui.add_folder("Shape"):
                handles += betas(server, state, key="shape", count=10)
            with server.gui.add_folder("Hand Pose"):
                handles += joint_xyz(server, state, key="hand_pose", joints=[(f"Joint {i}", i) for i in range(15)])
        elif name == "SKEL":
            with server.gui.add_folder("Shape"):
                handles += betas(server, state, key="shape", count=10)
            with server.gui.add_folder("Pose"):
                for label, idx, (lo, hi) in SKEL_POSE_DOFS:
                    handles.append(
                        add_slider(
                            server,
                            state,
                            label,
                            lo=lo,
                            hi=hi,
                            step=0.05,
                            initial=0.0,
                            key="body_pose",
                            indices=(idx,),
                        )
                    )
        elif name == "ANNY":
            with server.gui.add_folder("Phenotype"):
                for label in ANNY_PHENOTYPE_PARAMS:
                    handles.append(
                        add_slider(
                            server,
                            state,
                            label,
                            lo=0.0,
                            hi=1.0,
                            step=0.05,
                            initial=0.5,
                            key=label.lower(),
                            indices=(),
                        )
                    )
            with server.gui.add_folder("Pose"):
                handles += joint_xyz(
                    server,
                    state,
                    key="body_pose",
                    joints=ANNY_BODY_POSE_BONES,
                    max_joints=state.params["body_pose"].shape[0],
                )
        elif name == "MHR":
            with server.gui.add_folder("Shape"):
                handles += betas(server, state, key="shape", count=10)
            with server.gui.add_folder("Expression"):
                handles += betas(server, state, key="expression", count=15, prefix="psi", lo=-2.0, hi=2.0)
        elif name == "FLAME":
            with server.gui.add_folder("Shape"):
                handles += betas(server, state, key="shape", count=10)
            with server.gui.add_folder("Expression"):
                handles += betas(server, state, key="expression", count=10, prefix="psi", lo=-2.0, hi=2.0)
            with server.gui.add_folder("Head Pose"):
                handles += joint_xyz(server, state, key="head_pose", joints=FLAME_POSE_JOINTS, lo=-0.5, hi=0.5)
        elif name == "GarmentMeasurements":
            assert isinstance(state.model, GarmentMeasurements)
            with server.gui.add_folder("Shape"):
                handles += betas(server, state, key="shape", count=state.model.num_shape_components)
            with server.gui.add_folder("Body Pose"):
                handles += joint_xyz(
                    server,
                    state,
                    key="body_pose",
                    joints=GARMENT_MEASUREMENTS_BODY_POSE_JOINTS,
                    max_joints=state.params["body_pose"].shape[0],
                )
            with server.gui.add_folder("Head Pose"):
                handles += joint_xyz(
                    server,
                    state,
                    key="head_pose",
                    joints=GARMENT_MEASUREMENTS_HEAD_POSE_JOINTS,
                    max_joints=state.params["head_pose"].shape[0],
                )
        elif name == "SOMA":
            assert isinstance(state.model, SOMA)
            shape_default = float(state.params["shape"][0])
            with server.gui.add_folder("Identity"):
                handles += betas(
                    server,
                    state,
                    key="shape",
                    count=min(10, state.model.identity_dim),
                    prefix="identity",
                    lo=-1.0,
                    hi=1.0,
                    step=0.05,
                    initial=shape_default,
                )
            with server.gui.add_folder("Body Pose"):
                handles += joint_xyz(
                    server,
                    state,
                    key="body_pose",
                    joints=SOMA_BODY_POSE_JOINTS,
                    max_joints=state.params["body_pose"].shape[0],
                )
            with server.gui.add_folder("Head Pose"):
                handles += joint_xyz(
                    server,
                    state,
                    key="head_pose",
                    joints=SOMA_HEAD_POSE_JOINTS,
                    max_joints=state.params["head_pose"].shape[0],
                )
        else:
            raise ValueError(f"Unhandled model controls: {name}")

        reset_button(server, handles)
    return ModelControls(folder, handles)


def reset_button(server: viser.ViserServer, handles: list[SliderHandle]) -> None:
    button = server.gui.add_button("Reset")

    @button.on_click
    def _(_) -> None:
        for slider in handles:
            slider.handle.value = slider.initial


def apply_pose(state: ModelState, sliders: list[SliderHandle], pose_name: str) -> None:
    if pose_name == "tpose":
        preset = state.model.get_tpose(hands=state.hands) if state.model.has_hands else state.model.get_tpose()
    elif pose_name == "apose":
        preset = state.model.get_apose(hands=state.hands) if state.model.has_hands else state.model.get_apose()
    else:
        raise ValueError(f"Unknown pose: {pose_name}")
    updated_keys = set()
    for key in ("body_pose", "head_pose", "hand_pose", "global_rotation"):
        if key in preset and key in state.params:
            state.params[key] = np.asarray(preset[key]).copy()
            updated_keys.add(key)
    if state.display_global_rotation is not None:
        state.params["global_rotation"] = state.display_global_rotation.copy()
        updated_keys.add("global_rotation")
    for slider in sliders:
        if slider.key in updated_keys:
            slider.handle.value = float(state.params[slider.key][slider.indices])
    refit_to_floor(state)
    state.changed = True


def apply_hands(state: ModelState, sliders: list[SliderHandle], hands: Literal["default", "flat", "rest"]) -> None:
    preset = state.model.get_rest_pose(hands=hands)
    state.hands = hands
    state.params["hand_pose"] = np.asarray(preset["hand_pose"]).copy()
    for slider in sliders:
        if slider.key == "hand_pose":
            slider.handle.value = float(state.params[slider.key][slider.indices])
    state.changed = True


def refit_to_floor(state: ModelState) -> None:
    verts = state.model.forward_vertices(**state.params)
    state.params["global_translation"][1] -= float(verts[..., 1].min())


def set_gui_visible(handle: Any, visible: bool) -> None:
    handle.visible = visible
    if isinstance(handle, viser.GuiFolderHandle):
        for child in handle._children.values():
            set_gui_visible(child, visible)


if __name__ == "__main__":
    main()
