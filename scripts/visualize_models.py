# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "body-models",
#   "numpy>=2.4.1",
#   "torch>=2.9.1",
#   "viser>=0.2.32",
# ]
# [tool.uv.sources]
# body-models = { path = ".." }
# ///
"""Visualize all body models side by side with interactive controls.

Displays SMPL, SMPLX, SKEL, ANNY, and MHR models in a row with tabs for each
model's parameters. Uses viser for 3D visualization and GUI.

Usage:
    uv run scripts/visualize_models.py --smpl /path/to/SMPL_NEUTRAL.npz ...
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import viser

from body_models import ANNY, MHR, SKEL, SMPL, SMPLX, BodyModel

# Default paths relative to project root
ASSETS_DIR = Path(__file__).parent.parent / "tests" / "assets"
DEFAULT_PATHS = {
    "smpl": ASSETS_DIR / "smpl" / "model" / "SMPL_NEUTRAL.npz",
    "smplx": ASSETS_DIR / "smplx" / "model" / "SMPLX_NEUTRAL.npz",
    "skel": ASSETS_DIR / "skel" / "model",
    "anny": ASSETS_DIR / "anny" / "model",
    "mhr": ASSETS_DIR / "mhr" / "model",
}

AXES = ("X", "Y", "Z")


@dataclass
class ModelState:
    """State for a single model."""

    model: BodyModel
    params: dict[str, torch.Tensor]
    mesh_handle: viser.MeshHandle | None = None
    changed: bool = True
    x_offset: float = 0.0


@dataclass
class GuiState:
    """Holds all GUI handles and state."""

    model_states: dict[str, ModelState] = field(default_factory=dict)
    gui_handles: dict[str, dict[str, viser.GuiInputHandle]] = field(default_factory=dict)


def load_models(args: argparse.Namespace) -> dict[str, BodyModel]:
    """Load available models."""
    models = {}

    if args.smpl and Path(args.smpl).exists():
        print(f"Loading SMPL from {args.smpl}", flush=True)
        models["SMPL"] = SMPL(Path(args.smpl), gender=args.smpl_gender)

    if args.smplx and Path(args.smplx).exists():
        print(f"Loading SMPLX from {args.smplx}", flush=True)
        models["SMPLX"] = SMPLX(Path(args.smplx))

    if args.skel and Path(args.skel).exists():
        print(f"Loading SKEL from {args.skel}", flush=True)
        models["SKEL"] = SKEL(args.skel_gender, Path(args.skel))

    if args.anny and Path(args.anny).exists():
        print(f"Loading ANNY from {args.anny}", flush=True)
        models["ANNY"] = ANNY(Path(args.anny))

    if args.mhr and Path(args.mhr).exists():
        print(f"Loading MHR from {args.mhr}", flush=True)
        models["MHR"] = MHR(Path(args.mhr))

    return models


def _add_slider(
    server: viser.ViserServer,
    label: str,
    *,
    min: float,
    max: float,
    step: float,
    initial: float,
    on_update: Callable[[float], None],
) -> viser.GuiInputHandle:
    handle = server.gui.add_slider(label, min=min, max=max, step=step, initial_value=initial)

    @handle.on_update
    def _(event):
        on_update(event.target.value)

    return handle


def _set_param(
    state: ModelState,
    key: str,
    value: float,
    *,
    index: int | None = None,
    axis: int | None = None,
) -> None:
    tensor = state.params[key]
    if axis is None:
        if index is None:
            tensor[0] = value
        else:
            tensor[0, index] = value
    else:
        tensor[0, index, axis] = value
    state.changed = True


def _reset_handles(handles: dict[str, viser.GuiInputHandle], keys: Iterable[str], value: float) -> None:
    for key in keys:
        handles[key].value = value


def create_smpl_tab(
    server: viser.ViserServer,
    tab_group: viser.GuiTabGroupHandle,
    state: ModelState,
) -> dict[str, viser.GuiInputHandle]:
    """Create GUI controls for SMPL model."""
    handles: dict[str, viser.GuiInputHandle] = {}
    model = state.model
    assert isinstance(model, SMPL)

    num_shape = 10
    pose_joints = [
        ("Spine1", 2),
        ("Spine2", 5),
        ("Spine3", 8),
        ("Neck", 11),
        ("L_Shoulder", 15),
        ("R_Shoulder", 16),
    ]

    with tab_group.add_tab("SMPL", viser.Icon.USER):
        with server.gui.add_folder("Shape"):
            for i in range(num_shape):
                handles[f"shape_{i}"] = _add_slider(
                    server,
                    f"β{i}",
                    min=-3.0,
                    max=3.0,
                    step=0.1,
                    initial=0.0,
                    on_update=lambda value, idx=i: _set_param(state, "shape", value, index=idx),
                )

        with server.gui.add_folder("Body Pose"):
            for joint_name, joint_idx in pose_joints:
                for axis, axis_name in enumerate(AXES):
                    handles[f"pose_{joint_idx}_{axis}"] = _add_slider(
                        server,
                        f"{joint_name} {axis_name}",
                        min=-1.5,
                        max=1.5,
                        step=0.05,
                        initial=0.0,
                        on_update=lambda value, idx=joint_idx, ax=axis: _set_param(
                            state, "body_pose", value, index=idx, axis=ax
                        ),
                    )

        reset_btn = server.gui.add_button("Reset")

        @reset_btn.on_click
        def _(_):
            state.params = model.get_rest_pose()
            _reset_handles(handles, [f"shape_{i}" for i in range(num_shape)], 0.0)
            _reset_handles(
                handles,
                [f"pose_{joint_idx}_{axis}" for _, joint_idx in pose_joints for axis in range(3)],
                0.0,
            )
            state.changed = True

    return handles


def create_smplx_tab(
    server: viser.ViserServer,
    tab_group: viser.GuiTabGroupHandle,
    state: ModelState,
) -> dict[str, viser.GuiInputHandle]:
    """Create GUI controls for SMPLX model."""
    handles: dict[str, viser.GuiInputHandle] = {}
    model = state.model
    assert isinstance(model, SMPLX)

    num_shape = 10
    num_expr = 10
    pose_joints = [
        ("Spine1", 2),
        ("Spine2", 5),
        ("Spine3", 8),
        ("Neck", 11),
        ("L_Shoulder", 15),
        ("R_Shoulder", 16),
    ]

    with tab_group.add_tab("SMPLX", viser.Icon.USER):
        with server.gui.add_folder("Shape"):
            for i in range(num_shape):
                handles[f"shape_{i}"] = _add_slider(
                    server,
                    f"β{i}",
                    min=-3.0,
                    max=3.0,
                    step=0.1,
                    initial=0.0,
                    on_update=lambda value, idx=i: _set_param(state, "shape", value, index=idx),
                )

        with server.gui.add_folder("Expression"):
            for i in range(num_expr):
                handles[f"expr_{i}"] = _add_slider(
                    server,
                    f"ψ{i}",
                    min=-2.0,
                    max=2.0,
                    step=0.1,
                    initial=0.0,
                    on_update=lambda value, idx=i: _set_param(state, "expression", value, index=idx),
                )

        with server.gui.add_folder("Body Pose"):
            for joint_name, joint_idx in pose_joints:
                for axis, axis_name in enumerate(AXES):
                    handles[f"pose_{joint_idx}_{axis}"] = _add_slider(
                        server,
                        f"{joint_name} {axis_name}",
                        min=-1.5,
                        max=1.5,
                        step=0.05,
                        initial=0.0,
                        on_update=lambda value, idx=joint_idx, ax=axis: _set_param(
                            state, "body_pose", value, index=idx, axis=ax
                        ),
                    )

        reset_btn = server.gui.add_button("Reset")

        @reset_btn.on_click
        def _(_):
            state.params = model.get_rest_pose()
            _reset_handles(handles, [f"shape_{i}" for i in range(num_shape)], 0.0)
            _reset_handles(handles, [f"expr_{i}" for i in range(num_expr)], 0.0)
            _reset_handles(
                handles,
                [f"pose_{joint_idx}_{axis}" for _, joint_idx in pose_joints for axis in range(3)],
                0.0,
            )
            state.changed = True

    return handles


def create_skel_tab(
    server: viser.ViserServer,
    tab_group: viser.GuiTabGroupHandle,
    state: ModelState,
) -> dict[str, viser.GuiInputHandle]:
    """Create GUI controls for SKEL model."""
    handles: dict[str, viser.GuiInputHandle] = {}
    model = state.model
    assert isinstance(model, SKEL)

    num_shape = 10
    pose_dofs = [
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

    with tab_group.add_tab("SKEL", viser.Icon.USER):
        with server.gui.add_folder("Shape"):
            for i in range(num_shape):
                handles[f"shape_{i}"] = _add_slider(
                    server,
                    f"β{i}",
                    min=-3.0,
                    max=3.0,
                    step=0.1,
                    initial=0.0,
                    on_update=lambda value, idx=i: _set_param(state, "shape", value, index=idx),
                )

        with server.gui.add_folder("Pose"):
            for name, idx, (min_val, max_val) in pose_dofs:
                handles[f"pose_{idx}"] = _add_slider(
                    server,
                    name,
                    min=min_val,
                    max=max_val,
                    step=0.05,
                    initial=0.0,
                    on_update=lambda value, pose_idx=idx: _set_param(state, "pose", value, index=pose_idx),
                )

        reset_btn = server.gui.add_button("Reset")

        @reset_btn.on_click
        def _(_):
            state.params = model.get_rest_pose()
            _reset_handles(handles, [f"shape_{i}" for i in range(num_shape)], 0.0)
            _reset_handles(handles, [f"pose_{idx}" for _, idx, _ in pose_dofs], 0.0)
            state.changed = True

    return handles


def create_anny_tab(
    server: viser.ViserServer,
    tab_group: viser.GuiTabGroupHandle,
    state: ModelState,
) -> dict[str, viser.GuiInputHandle]:
    """Create GUI controls for ANNY model."""
    handles: dict[str, viser.GuiInputHandle] = {}
    model = state.model
    assert isinstance(model, ANNY)

    phenotype_params = ["Gender", "Age", "Muscle", "Weight", "Height", "Proportions"]
    pose_bones = [
        ("Spine", 1),
        ("Spine1", 2),
        ("Spine2", 3),
        ("Neck", 4),
        ("L Shoulder", 8),
        ("L Arm", 9),
        ("R Shoulder", 13),
        ("R Arm", 14),
        ("L UpLeg", 18),
        ("R UpLeg", 23),
    ]

    with tab_group.add_tab("ANNY", viser.Icon.USER):
        with server.gui.add_folder("Phenotype"):
            for param in phenotype_params:
                param_key = param.lower()
                handles[param_key] = _add_slider(
                    server,
                    param,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    initial=0.5,
                    on_update=lambda value, key=param_key: _set_param(state, key, value),
                )

        with server.gui.add_folder("Pose"):
            for bone_name, bone_idx in pose_bones:
                if bone_idx >= model.num_joints:
                    continue
                for axis, axis_name in enumerate(AXES):
                    handles[f"pose_{bone_idx}_{axis}"] = _add_slider(
                        server,
                        f"{bone_name} {axis_name}",
                        min=-1.5,
                        max=1.5,
                        step=0.05,
                        initial=0.0,
                        on_update=lambda value, idx=bone_idx, ax=axis: _set_param(
                            state, "pose", value, index=idx, axis=ax
                        ),
                    )

        reset_btn = server.gui.add_button("Reset")

        @reset_btn.on_click
        def _(_):
            state.params = model.get_rest_pose()
            _reset_handles(handles, [param.lower() for param in phenotype_params], 0.5)
            _reset_handles(
                handles,
                [
                    f"pose_{bone_idx}_{axis}"
                    for _, bone_idx in pose_bones
                    if bone_idx < model.num_joints
                    for axis in range(3)
                ],
                0.0,
            )
            state.changed = True

    return handles


def create_mhr_tab(
    server: viser.ViserServer,
    tab_group: viser.GuiTabGroupHandle,
    state: ModelState,
) -> dict[str, viser.GuiInputHandle]:
    """Create GUI controls for MHR model."""
    handles: dict[str, viser.GuiInputHandle] = {}
    model = state.model
    assert isinstance(model, MHR)

    num_shape = 10
    num_expr = 15

    with tab_group.add_tab("MHR", viser.Icon.USER):
        with server.gui.add_folder("Shape"):
            for i in range(num_shape):
                handles[f"shape_{i}"] = _add_slider(
                    server,
                    f"β{i}",
                    min=-3.0,
                    max=3.0,
                    step=0.1,
                    initial=0.0,
                    on_update=lambda value, idx=i: _set_param(state, "shape", value, index=idx),
                )

        with server.gui.add_folder("Expression"):
            for i in range(num_expr):
                handles[f"expr_{i}"] = _add_slider(
                    server,
                    f"ψ{i}",
                    min=-2.0,
                    max=2.0,
                    step=0.1,
                    initial=0.0,
                    on_update=lambda value, idx=i: _set_param(state, "expression", value, index=idx),
                )

        reset_btn = server.gui.add_button("Reset")

        @reset_btn.on_click
        def _(_):
            state.params = model.get_rest_pose()
            _reset_handles(handles, [f"shape_{i}" for i in range(num_shape)], 0.0)
            _reset_handles(handles, [f"expr_{i}" for i in range(num_expr)], 0.0)
            state.changed = True

    return handles


def update_mesh(server: viser.ViserServer, name: str, state: ModelState) -> None:
    with torch.no_grad():
        vertices = state.model.forward_vertices(**state.params)

    verts_np = vertices[0].cpu().numpy()
    faces_np = state.model.faces.cpu().numpy()

    # Triangulate quads
    if faces_np.shape[1] == 4:
        tri1 = faces_np[:, [0, 1, 2]]
        tri2 = faces_np[:, [0, 2, 3]]
        faces_np = np.concatenate([tri1, tri2], axis=0)

    position = (state.x_offset, 0.0, 0.0)
    if state.mesh_handle is None:
        state.mesh_handle = server.scene.add_mesh_simple(
            f"/{name}",
            vertices=verts_np,
            faces=faces_np,
            color=(90, 200, 255),
            position=position,
        )
    else:
        state.mesh_handle.vertices = verts_np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smpl", type=str, default=str(DEFAULT_PATHS["smpl"]), help="Path to SMPL model file (.npz)")
    parser.add_argument(
        "--smpl-gender",
        type=str,
        default="neutral",
        choices=["neutral", "male", "female"],
        help="Gender for SMPL model",
    )
    parser.add_argument(
        "--smplx", type=str, default=str(DEFAULT_PATHS["smplx"]), help="Path to SMPLX model file (.npz)"
    )
    parser.add_argument("--skel", type=str, default=str(DEFAULT_PATHS["skel"]), help="Path to SKEL model directory")
    parser.add_argument(
        "--skel-gender", type=str, default="male", choices=["male", "female"], help="Gender for SKEL model"
    )
    parser.add_argument("--anny", type=str, default=str(DEFAULT_PATHS["anny"]), help="Path to ANNY model directory")
    parser.add_argument("--mhr", type=str, default=str(DEFAULT_PATHS["mhr"]), help="Path to MHR model directory")
    args = parser.parse_args()

    server = viser.ViserServer()
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/grid", position=(0.0, 0.0, 0.0), plane="xz")
    server.gui.configure_theme(control_layout="fixed", control_width="large")

    models = load_models(args)
    if not models:
        print("No models loaded! Please provide at least one model path.", flush=True)
        print("Example: uv run scripts/visualize_models.py --smplx /path/to/SMPLX_NEUTRAL.npz", flush=True)
        return

    print(f"Loaded {len(models)} models: {list(models.keys())}", flush=True)

    gui_state = GuiState()
    x_positions = {
        "SMPL": -2.0,
        "SMPLX": -1.0,
        "SKEL": 0.0,
        "ANNY": 1.0,
        "MHR": 2.0,
    }

    for name, model in models.items():
        params = model.get_rest_pose()
        gui_state.model_states[name] = ModelState(
            model=model,
            params=params,
            x_offset=x_positions.get(name, 0.0),
        )

    tab_group = server.gui.add_tab_group()
    tab_creators = {
        "SMPL": create_smpl_tab,
        "SMPLX": create_smplx_tab,
        "SKEL": create_skel_tab,
        "ANNY": create_anny_tab,
        "MHR": create_mhr_tab,
    }
    for name, state in gui_state.model_states.items():
        if name in tab_creators:
            gui_state.gui_handles[name] = tab_creators[name](server, tab_group, state)

    for name, state in gui_state.model_states.items():
        with torch.no_grad():
            verts = state.model.forward_vertices(**state.params)
            label_y = float(verts[..., 1].max()) + 0.1  # 10cm above head
        server.scene.add_label(
            f"/labels/{name}",
            text=name,
            position=(state.x_offset, label_y, 0.0),
        )

    print("\nServer running at http://localhost:8080", flush=True)

    while True:
        time.sleep(0.02)

        for name, state in gui_state.model_states.items():
            if state.changed:
                state.changed = False
                update_mesh(server, name, state)


if __name__ == "__main__":
    main()
