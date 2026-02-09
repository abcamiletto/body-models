# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "body-models",
#   "numpy>=2.4.1",
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
import viser

from body_models.anny.numpy import ANNY
from body_models.base import BodyModel
from body_models.flame.numpy import FLAME
from body_models.mhr.numpy import MHR
from body_models.skel.numpy import SKEL
from body_models.smpl.numpy import SMPL
from body_models.smplx.numpy import SMPLX

# Default paths relative to project root
ASSETS_DIR = Path(__file__).parent.parent / "tests" / "assets"
DEFAULT_PATHS = {
    "smpl": ASSETS_DIR / "smpl" / "model" / "SMPL_NEUTRAL.npz",
    "smplx": ASSETS_DIR / "smplx" / "model" / "SMPLX_NEUTRAL.npz",
    "skel": ASSETS_DIR / "skel" / "model",
    "anny": ASSETS_DIR / "anny" / "model",
    "mhr": ASSETS_DIR / "mhr" / "model",
    "flame": ASSETS_DIR / "flame" / "model" / "FLAME2023",
}

AXES = ("X", "Y", "Z")

# Configuration constants for pose/shape controls
SMPL_POSE_JOINTS = [
    ("Spine1", 2),
    ("Spine2", 5),
    ("Spine3", 8),
    ("Neck", 11),
    ("L_Shoulder", 15),
    ("R_Shoulder", 16),
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

FLAME_POSE_JOINTS = [
    ("Neck", 0),
    ("Jaw", 1),
    ("L_Eye", 2),
    ("R_Eye", 3),
]

ANNY_PHENOTYPE_PARAMS = ["Gender", "Age", "Muscle", "Weight", "Height", "Proportions"]
ANNY_POSE_BONES = [
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

# Pastel colors for each model (RGB tuples)
MODEL_COLORS: dict[str, tuple[int, int, int]] = {
    "SMPL": (173, 216, 230),  # Light blue
    "SMPLX": (255, 182, 193),  # Light pink
    "SKEL": (144, 238, 144),  # Light green
    "ANNY": (255, 218, 185),  # Peach
    "MHR": (221, 160, 221),  # Plum/lavender
    "FLAME": (255, 239, 186),  # Light yellow/cream
}


@dataclass
class ResetGroup:
    """Group of handles to reset together."""

    keys: list[str]
    value: float


@dataclass
class ModelState:
    """State for a single model."""

    model: BodyModel
    params: dict[str, np.ndarray]
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

    if args.flame and Path(args.flame).exists():
        print(f"Loading FLAME from {args.flame}", flush=True)
        models["FLAME"] = FLAME(Path(args.flame))

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


def create_indexed_sliders(
    server: viser.ViserServer,
    state: ModelState,
    handles: dict[str, viser.GuiInputHandle],
    *,
    param_key: str,
    count: int,
    prefix: str,
    label_prefix: str,
    min_val: float,
    max_val: float,
    step: float,
    initial: float,
) -> list[str]:
    """Create sliders for indexed parameters (shape betas, expressions)."""
    keys = []
    for i in range(count):
        key = f"{prefix}_{i}"
        keys.append(key)
        handles[key] = _add_slider(
            server,
            f"{label_prefix}{i}",
            min=min_val,
            max=max_val,
            step=step,
            initial=initial,
            on_update=lambda value, idx=i: _set_param(state, param_key, value, index=idx),
        )
    return keys


def create_joint_pose_sliders(
    server: viser.ViserServer,
    state: ModelState,
    handles: dict[str, viser.GuiInputHandle],
    *,
    param_key: str,
    joints: list[tuple[str, int]],
    min_val: float = -1.5,
    max_val: float = 1.5,
    step: float = 0.05,
    max_joints: int | None = None,
) -> list[str]:
    """Create XYZ sliders for each joint (axis-angle pose)."""
    keys = []
    for joint_name, joint_idx in joints:
        if max_joints is not None and joint_idx >= max_joints:
            continue
        for axis, axis_name in enumerate(AXES):
            key = f"pose_{joint_idx}_{axis}"
            keys.append(key)
            handles[key] = _add_slider(
                server,
                f"{joint_name} {axis_name}",
                min=min_val,
                max=max_val,
                step=step,
                initial=0.0,
                on_update=lambda value, idx=joint_idx, ax=axis: _set_param(state, param_key, value, index=idx, axis=ax),
            )
    return keys


def create_dof_sliders(
    server: viser.ViserServer,
    state: ModelState,
    handles: dict[str, viser.GuiInputHandle],
    *,
    param_key: str,
    dofs: list[tuple[str, int, tuple[float, float]]],
    step: float = 0.05,
) -> list[str]:
    """Create sliders for flat DOF vector (SKEL-style)."""
    keys = []
    for name, idx, (min_val, max_val) in dofs:
        key = f"pose_{idx}"
        keys.append(key)
        handles[key] = _add_slider(
            server,
            name,
            min=min_val,
            max=max_val,
            step=step,
            initial=0.0,
            on_update=lambda value, pose_idx=idx: _set_param(state, param_key, value, index=pose_idx),
        )
    return keys


def create_phenotype_sliders(
    server: viser.ViserServer,
    state: ModelState,
    handles: dict[str, viser.GuiInputHandle],
    *,
    params: list[str],
    min_val: float = 0.0,
    max_val: float = 1.0,
    step: float = 0.05,
    initial: float = 0.5,
) -> list[str]:
    """Create sliders for phenotype parameters (0-1 range)."""
    keys = []
    for param in params:
        param_key = param.lower()
        keys.append(param_key)
        handles[param_key] = _add_slider(
            server,
            param,
            min=min_val,
            max=max_val,
            step=step,
            initial=initial,
            on_update=lambda value, key=param_key: _set_param(state, key, value),
        )
    return keys


def add_reset_button(
    server: viser.ViserServer,
    state: ModelState,
    handles: dict[str, viser.GuiInputHandle],
    reset_groups: list[ResetGroup],
) -> None:
    """Add a reset button that resets all specified handle groups."""
    reset_btn = server.gui.add_button("Reset")

    @reset_btn.on_click
    def _(_):
        state.params = state.model.get_rest_pose()
        for group in reset_groups:
            _reset_handles(handles, group.keys, group.value)
        state.changed = True


def create_smpl_family_tab(
    server: viser.ViserServer,
    tab_group: viser.GuiTabGroupHandle,
    state: ModelState,
    *,
    tab_name: str,
    num_shape: int = 10,
    num_expr: int | None = None,
    pose_joints: list[tuple[str, int]],
) -> dict[str, viser.GuiInputHandle]:
    """Shared implementation for SMPL/SMPLX tabs with optional expression folder."""
    handles: dict[str, viser.GuiInputHandle] = {}
    reset_groups: list[ResetGroup] = []

    with tab_group.add_tab(tab_name, viser.Icon.USER):
        with server.gui.add_folder("Shape"):
            shape_keys = create_indexed_sliders(
                server,
                state,
                handles,
                param_key="shape",
                count=num_shape,
                prefix="shape",
                label_prefix="β",
                min_val=-3.0,
                max_val=3.0,
                step=0.1,
                initial=0.0,
            )
            reset_groups.append(ResetGroup(keys=shape_keys, value=0.0))

        if num_expr is not None:
            with server.gui.add_folder("Expression"):
                expr_keys = create_indexed_sliders(
                    server,
                    state,
                    handles,
                    param_key="expression",
                    count=num_expr,
                    prefix="expr",
                    label_prefix="ψ",
                    min_val=-2.0,
                    max_val=2.0,
                    step=0.1,
                    initial=0.0,
                )
                reset_groups.append(ResetGroup(keys=expr_keys, value=0.0))

        with server.gui.add_folder("Body Pose"):
            pose_keys = create_joint_pose_sliders(
                server,
                state,
                handles,
                param_key="body_pose",
                joints=pose_joints,
            )
            reset_groups.append(ResetGroup(keys=pose_keys, value=0.0))

        add_reset_button(server, state, handles, reset_groups)

    return handles


def create_smpl_tab(
    server: viser.ViserServer,
    tab_group: viser.GuiTabGroupHandle,
    state: ModelState,
) -> dict[str, viser.GuiInputHandle]:
    """Create GUI controls for SMPL model."""
    return create_smpl_family_tab(server, tab_group, state, tab_name="SMPL", pose_joints=SMPL_POSE_JOINTS)


def create_smplx_tab(
    server: viser.ViserServer,
    tab_group: viser.GuiTabGroupHandle,
    state: ModelState,
) -> dict[str, viser.GuiInputHandle]:
    """Create GUI controls for SMPLX model."""
    return create_smpl_family_tab(server, tab_group, state, tab_name="SMPLX", num_expr=10, pose_joints=SMPL_POSE_JOINTS)


def create_skel_tab(
    server: viser.ViserServer,
    tab_group: viser.GuiTabGroupHandle,
    state: ModelState,
) -> dict[str, viser.GuiInputHandle]:
    """Create GUI controls for SKEL model."""
    handles: dict[str, viser.GuiInputHandle] = {}
    reset_groups: list[ResetGroup] = []
    num_shape = 10

    with tab_group.add_tab("SKEL", viser.Icon.USER):
        with server.gui.add_folder("Shape"):
            shape_keys = create_indexed_sliders(
                server,
                state,
                handles,
                param_key="shape",
                count=num_shape,
                prefix="shape",
                label_prefix="β",
                min_val=-3.0,
                max_val=3.0,
                step=0.1,
                initial=0.0,
            )
            reset_groups.append(ResetGroup(keys=shape_keys, value=0.0))

        with server.gui.add_folder("Pose"):
            pose_keys = create_dof_sliders(server, state, handles, param_key="pose", dofs=SKEL_POSE_DOFS)
            reset_groups.append(ResetGroup(keys=pose_keys, value=0.0))

        add_reset_button(server, state, handles, reset_groups)

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
    reset_groups: list[ResetGroup] = []

    with tab_group.add_tab("ANNY", viser.Icon.USER):
        with server.gui.add_folder("Phenotype"):
            pheno_keys = create_phenotype_sliders(server, state, handles, params=ANNY_PHENOTYPE_PARAMS)
            reset_groups.append(ResetGroup(keys=pheno_keys, value=0.5))

        with server.gui.add_folder("Pose"):
            pose_keys = create_joint_pose_sliders(
                server,
                state,
                handles,
                param_key="pose",
                joints=ANNY_POSE_BONES,
                max_joints=model.num_joints,
            )
            reset_groups.append(ResetGroup(keys=pose_keys, value=0.0))

        add_reset_button(server, state, handles, reset_groups)

    return handles


def create_mhr_tab(
    server: viser.ViserServer,
    tab_group: viser.GuiTabGroupHandle,
    state: ModelState,
) -> dict[str, viser.GuiInputHandle]:
    """Create GUI controls for MHR model."""
    handles: dict[str, viser.GuiInputHandle] = {}
    reset_groups: list[ResetGroup] = []

    with tab_group.add_tab("MHR", viser.Icon.USER):
        with server.gui.add_folder("Shape"):
            shape_keys = create_indexed_sliders(
                server,
                state,
                handles,
                param_key="shape",
                count=10,
                prefix="shape",
                label_prefix="β",
                min_val=-3.0,
                max_val=3.0,
                step=0.1,
                initial=0.0,
            )
            reset_groups.append(ResetGroup(keys=shape_keys, value=0.0))

        with server.gui.add_folder("Expression"):
            expr_keys = create_indexed_sliders(
                server,
                state,
                handles,
                param_key="expression",
                count=15,
                prefix="expr",
                label_prefix="ψ",
                min_val=-2.0,
                max_val=2.0,
                step=0.1,
                initial=0.0,
            )
            reset_groups.append(ResetGroup(keys=expr_keys, value=0.0))

        add_reset_button(server, state, handles, reset_groups)

    return handles


def create_flame_tab(
    server: viser.ViserServer,
    tab_group: viser.GuiTabGroupHandle,
    state: ModelState,
) -> dict[str, viser.GuiInputHandle]:
    """Create GUI controls for FLAME model."""
    handles: dict[str, viser.GuiInputHandle] = {}
    reset_groups: list[ResetGroup] = []

    with tab_group.add_tab("FLAME", viser.Icon.USER):
        with server.gui.add_folder("Shape"):
            shape_keys = create_indexed_sliders(
                server,
                state,
                handles,
                param_key="shape",
                count=10,
                prefix="shape",
                label_prefix="β",
                min_val=-3.0,
                max_val=3.0,
                step=0.1,
                initial=0.0,
            )
            reset_groups.append(ResetGroup(keys=shape_keys, value=0.0))

        with server.gui.add_folder("Expression"):
            expr_keys = create_indexed_sliders(
                server,
                state,
                handles,
                param_key="expression",
                count=10,
                prefix="expr",
                label_prefix="ψ",
                min_val=-2.0,
                max_val=2.0,
                step=0.1,
                initial=0.0,
            )
            reset_groups.append(ResetGroup(keys=expr_keys, value=0.0))

        with server.gui.add_folder("Head Pose"):
            pose_keys = create_joint_pose_sliders(
                server,
                state,
                handles,
                param_key="head_pose",
                joints=FLAME_POSE_JOINTS,
                min_val=-0.5,
                max_val=0.5,
            )
            reset_groups.append(ResetGroup(keys=pose_keys, value=0.0))

        add_reset_button(server, state, handles, reset_groups)

    return handles


def update_mesh(server: viser.ViserServer, name: str, state: ModelState) -> None:
    vertices = state.model.forward_vertices(**state.params)

    verts_np = vertices[0]
    faces_np = state.model.faces

    # Triangulate quads
    if faces_np.shape[1] == 4:
        tri1 = faces_np[:, [0, 1, 2]]
        tri2 = faces_np[:, [0, 2, 3]]
        faces_np = np.concatenate([tri1, tri2], axis=0)

    position = (state.x_offset, 0.0, 0.0)
    color = MODEL_COLORS.get(name, (200, 200, 200))
    if state.mesh_handle is None:
        state.mesh_handle = server.scene.add_mesh_simple(
            f"/{name}",
            vertices=verts_np,
            faces=faces_np,
            color=color,
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
    parser.add_argument("--flame", type=str, default=str(DEFAULT_PATHS["flame"]), help="Path to FLAME model directory")
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
        "SMPL": -2.5,
        "SMPLX": -1.5,
        "SKEL": -0.5,
        "ANNY": 0.5,
        "MHR": 1.5,
        "FLAME": 2.5,
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
        "FLAME": create_flame_tab,
    }
    for name, state in gui_state.model_states.items():
        if name in tab_creators:
            gui_state.gui_handles[name] = tab_creators[name](server, tab_group, state)

    for name, state in gui_state.model_states.items():
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
