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

Default asset paths point at ``tests/assets/<model>/model/...``; SOMA falls back
to the auto-downloaded body-models cache. Adjust the constants below to relocate.

Usage:
    uv run scripts/visualize_models.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser

from body_models.anny.numpy import ANNY
from body_models.base import BodyModel
from body_models.flame.numpy import FLAME
from body_models.g1.numpy import G1
from body_models.garment_measurements.numpy import GarmentMeasurements
from body_models.mhr.numpy import MHR
from body_models.skel.numpy import SKEL
from body_models.smpl.numpy import SMPL
from body_models.smplx.numpy import SMPLX
from body_models.soma.numpy import SOMA

# ── Asset locations ──────────────────────────────────────────────────────────
ASSETS_DIR = Path(__file__).parent.parent / "tests" / "assets"
SMPL_PATH = ASSETS_DIR / "smpl/model/SMPL_NEUTRAL.npz"
SMPLX_PATH = ASSETS_DIR / "smplx/model/SMPLX_NEUTRAL.npz"
SKEL_PATH = ASSETS_DIR / "skel/model"
ANNY_PATH = ASSETS_DIR / "anny/model"
MHR_PATH = ASSETS_DIR / "mhr/model"
FLAME_PATH = ASSETS_DIR / "flame/model/FLAME_NEUTRAL.pkl"
GARMENT_MEASUREMENTS_PATH = ASSETS_DIR / "garment_measurements/model"
G1_PATH = ASSETS_DIR / "g1/model"
SOMA_PATH: Path | None = None  # None → load from body-models cache
ANNY_DISPLAY_ROTATION_X = -np.pi / 2

# ── Per-tab joint configurations ─────────────────────────────────────────────
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

FLAME_POSE_JOINTS = [("Neck", 0), ("Jaw", 1), ("L_Eye", 2), ("R_Eye", 3)]

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

GARMENT_MEASUREMENTS_POSE_JOINTS = [
    ("Spine1", 1),
    ("Spine2", 2),
    ("Chest", 3),
    ("Neck1", 4),
    ("Head", 6),
    ("L Clavicle", 9),
    ("L UpperArm", 10),
    ("R Clavicle", 30),
    ("R UpperArm", 31),
    ("L Thigh", 51),
    ("R Thigh", 55),
]

SOMA_POSE_JOINTS = [
    ("Spine1", 1),
    ("Spine2", 2),
    ("Chest", 3),
    ("Neck1", 4),
    ("Head", 6),
    ("L Shoulder", 11),
    ("L Arm", 12),
    ("L ForeArm", 13),
    ("R Shoulder", 39),
    ("R Arm", 40),
    ("R ForeArm", 41),
    ("L Leg", 67),
    ("R Leg", 72),
]

# qpos joint indices into model.qpos_joint_names. With rotation_type="hinge"
# body_pose stores one scalar angle per qpos entry, so each joint = one slider.
G1_POSE_JOINTS = [
    ("L Hip Pitch", 0),
    ("L Hip Roll", 1),
    ("L Hip Yaw", 2),
    ("L Knee", 3),
    ("R Hip Pitch", 6),
    ("R Hip Roll", 7),
    ("R Hip Yaw", 8),
    ("R Knee", 9),
    ("Waist Yaw", 12),
    ("Waist Pitch", 14),
    ("L Shoulder Pitch", 15),
    ("L Shoulder Roll", 16),
    ("L Elbow", 18),
    ("R Shoulder Pitch", 22),
    ("R Shoulder Roll", 23),
    ("R Elbow", 25),
]

# Grid layout: split models across rows on the xz ground plane.
GRID_COLS = 5  # max models per row; with 9 models this gives 5 + 4
GRID_SPACING_X = 1.8
GRID_SPACING_Z = 1.8

MODEL_COLORS: dict[str, tuple[int, int, int]] = {
    "SMPL": (173, 216, 230),
    "SMPLX": (255, 182, 193),
    "SKEL": (144, 238, 144),
    "ANNY": (255, 218, 185),
    "MHR": (221, 160, 221),
    "FLAME": (255, 239, 186),
    "GarmentMeasurements": (176, 224, 230),
    "SOMA": (250, 200, 200),
    "G1": (200, 200, 220),
}


# ── State ────────────────────────────────────────────────────────────────────
@dataclass
class ModelState:
    model: BodyModel
    params: dict[str, np.ndarray]
    faces: np.ndarray
    color: tuple[int, int, int]
    x_offset: float
    y_offset: float
    z_offset: float
    mesh_handle: viser.MeshHandle | None = None
    changed: bool = True


SliderHandle = tuple[viser.GuiInputHandle, float]  # (handle, initial value)


# ── Slider primitives ────────────────────────────────────────────────────────
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
    """Create a slider that writes ``state.params[key][indices] = value`` on update."""
    handle = server.gui.add_slider(label, min=lo, max=hi, step=step, initial_value=initial)

    @handle.on_update
    def _(event):
        state.params[key][indices] = event.target.value
        state.changed = True

    return handle, initial


def betas(server, state, *, key, count, prefix="β", lo=-3.0, hi=3.0, step=0.1, initial=0.0) -> list[SliderHandle]:
    """Indexed sliders writing into ``state.params[key][0, i]``."""
    return [
        add_slider(server, state, f"{prefix}{i}", lo=lo, hi=hi, step=step, initial=initial, key=key, indices=(0, i))
        for i in range(count)
    ]


def joint_xyz(server, state, *, key, joints, lo=-1.5, hi=1.5, step=0.05, max_joints=None) -> list[SliderHandle]:
    """X/Y/Z axis-angle sliders per joint, writing into ``state.params[key][0, joint, axis]``."""
    handles: list[SliderHandle] = []
    for name, idx in joints:
        if max_joints is not None and idx >= max_joints:
            continue
        for ax, axn in enumerate("XYZ"):
            handles.append(
                add_slider(
                    server, state, f"{name} {axn}", lo=lo, hi=hi, step=step, initial=0.0, key=key, indices=(0, idx, ax)
                )
            )
    return handles


def reset_button(server: viser.ViserServer, handles: list[SliderHandle]) -> None:
    """Reset button: setting ``handle.value`` fires on_update and writes the initial back."""
    btn = server.gui.add_button("Reset")

    @btn.on_click
    def _(_):
        for handle, initial in handles:
            handle.value = initial


# ── Per-model tab builders ───────────────────────────────────────────────────
def smpl_tab(server, tabs, state, *, name="SMPL", with_expression=False) -> None:
    handles: list[SliderHandle] = []
    with tabs.add_tab(name, viser.Icon.USER):
        with server.gui.add_folder("Shape"):
            handles += betas(server, state, key="shape", count=10)
        if with_expression:
            with server.gui.add_folder("Expression"):
                handles += betas(server, state, key="expression", count=10, prefix="ψ", lo=-2.0, hi=2.0)
        with server.gui.add_folder("Body Pose"):
            handles += joint_xyz(server, state, key="body_pose", joints=SMPL_POSE_JOINTS)
        reset_button(server, handles)


def smplx_tab(server, tabs, state) -> None:
    smpl_tab(server, tabs, state, name="SMPLX", with_expression=True)


def skel_tab(server, tabs, state) -> None:
    handles: list[SliderHandle] = []
    with tabs.add_tab("SKEL", viser.Icon.USER):
        with server.gui.add_folder("Shape"):
            handles += betas(server, state, key="shape", count=10)
        with server.gui.add_folder("Pose"):
            for label, idx, (lo, hi) in SKEL_POSE_DOFS:
                handles.append(
                    add_slider(server, state, label, lo=lo, hi=hi, step=0.05, initial=0.0, key="pose", indices=(0, idx))
                )
        reset_button(server, handles)


def anny_tab(server, tabs, state) -> None:
    handles: list[SliderHandle] = []
    with tabs.add_tab("ANNY", viser.Icon.USER):
        with server.gui.add_folder("Phenotype"):
            for label in ANNY_PHENOTYPE_PARAMS:
                handles.append(
                    add_slider(
                        server, state, label, lo=0.0, hi=1.0, step=0.05, initial=0.5, key=label.lower(), indices=(0,)
                    )
                )
        with server.gui.add_folder("Pose"):
            handles += joint_xyz(server, state, key="pose", joints=ANNY_POSE_BONES, max_joints=state.model.num_joints)
        reset_button(server, handles)


def mhr_tab(server, tabs, state) -> None:
    handles: list[SliderHandle] = []
    with tabs.add_tab("MHR", viser.Icon.USER):
        with server.gui.add_folder("Shape"):
            handles += betas(server, state, key="shape", count=10)
        with server.gui.add_folder("Expression"):
            handles += betas(server, state, key="expression", count=15, prefix="ψ", lo=-2.0, hi=2.0)
        reset_button(server, handles)


def flame_tab(server, tabs, state) -> None:
    handles: list[SliderHandle] = []
    with tabs.add_tab("FLAME", viser.Icon.USER):
        with server.gui.add_folder("Shape"):
            handles += betas(server, state, key="shape", count=10)
        with server.gui.add_folder("Expression"):
            handles += betas(server, state, key="expression", count=10, prefix="ψ", lo=-2.0, hi=2.0)
        with server.gui.add_folder("Head Pose"):
            handles += joint_xyz(server, state, key="head_pose", joints=FLAME_POSE_JOINTS, lo=-0.5, hi=0.5)
        reset_button(server, handles)


def garment_measurements_tab(server, tabs, state) -> None:
    handles: list[SliderHandle] = []
    with tabs.add_tab("GarmentMeasurements", viser.Icon.USER):
        with server.gui.add_folder("Shape"):
            handles += betas(server, state, key="shape", count=state.model.num_shape_components)
        with server.gui.add_folder("Pose"):
            handles += joint_xyz(
                server, state, key="pose", joints=GARMENT_MEASUREMENTS_POSE_JOINTS, max_joints=state.model.num_joints
            )
        reset_button(server, handles)


def soma_tab(server, tabs, state) -> None:
    handles: list[SliderHandle] = []
    identity_default = float(state.params["identity"][0, 0])
    with tabs.add_tab("SOMA", viser.Icon.USER):
        with server.gui.add_folder("Identity"):
            handles += betas(
                server,
                state,
                key="identity",
                count=min(10, state.model.identity_dim),
                prefix="ι",
                lo=-1.0,
                hi=1.0,
                step=0.05,
                initial=identity_default,
            )
        with server.gui.add_folder("Pose"):
            handles += joint_xyz(server, state, key="pose", joints=SOMA_POSE_JOINTS, max_joints=state.model.num_joints)
        reset_button(server, handles)


def g1_tab(server, tabs, state) -> None:
    handles: list[SliderHandle] = []
    with tabs.add_tab("G1", viser.Icon.USER):
        with server.gui.add_folder("Hinge Pose"):
            for label, qpos_idx in G1_POSE_JOINTS:
                if qpos_idx >= len(state.model.qpos_joint_indices):
                    continue
                lo, hi = (float(x) for x in state.model.qpos_joint_limits[qpos_idx])
                # qpos limits can be ±inf for free joints — clamp to a usable range.
                if not np.isfinite(lo) or not np.isfinite(hi):
                    lo, hi = -np.pi, np.pi
                handles.append(
                    add_slider(
                        server,
                        state,
                        label,
                        lo=lo,
                        hi=hi,
                        step=0.02,
                        initial=0.0,
                        key="body_pose",
                        indices=(0, qpos_idx, 0),
                    )
                )
        reset_button(server, handles)


TAB_BUILDERS = {
    "SMPL": smpl_tab,
    "SMPLX": smplx_tab,
    "SKEL": skel_tab,
    "ANNY": anny_tab,
    "MHR": mhr_tab,
    "FLAME": flame_tab,
    "GarmentMeasurements": garment_measurements_tab,
    "SOMA": soma_tab,
    "G1": g1_tab,
}


# ── Mesh & main loop ─────────────────────────────────────────────────────────
def triangulate(faces: np.ndarray) -> np.ndarray:
    if faces.shape[1] == 3:
        return faces
    if faces.shape[1] == 4:
        return np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)
    raise ValueError(f"Unsupported face arity: {faces.shape[1]}")


def update_mesh(server: viser.ViserServer, name: str, state: ModelState) -> None:
    verts = state.model.forward_vertices(**state.params)[0]
    if state.mesh_handle is None:
        state.mesh_handle = server.scene.add_mesh_simple(
            f"/{name}",
            vertices=verts,
            faces=state.faces,
            color=state.color,
            position=(state.x_offset, state.y_offset, state.z_offset),
        )
    else:
        state.mesh_handle.vertices = verts


def load_models() -> dict[str, BodyModel]:
    print(f"Loading SMPL from {SMPL_PATH}", flush=True)
    smpl = SMPL(SMPL_PATH, gender="neutral")
    print(f"Loading SMPLX from {SMPLX_PATH}", flush=True)
    smplx = SMPLX(SMPLX_PATH)
    print(f"Loading SKEL from {SKEL_PATH}", flush=True)
    skel = SKEL(SKEL_PATH, "male")
    print(f"Loading ANNY from {ANNY_PATH}", flush=True)
    anny = ANNY(ANNY_PATH)
    print(f"Loading MHR from {MHR_PATH}", flush=True)
    mhr = MHR(MHR_PATH)
    print(f"Loading FLAME from {FLAME_PATH}", flush=True)
    flame = FLAME(FLAME_PATH)
    print(f"Loading GarmentMeasurements from {GARMENT_MEASUREMENTS_PATH}", flush=True)
    gm = GarmentMeasurements(GARMENT_MEASUREMENTS_PATH)
    print(f"Loading SOMA from {SOMA_PATH or '<cache>'}", flush=True)
    soma = SOMA(model_path=SOMA_PATH)
    print(f"Loading G1 from {G1_PATH}", flush=True)
    # Hinge parametrization: one scalar angle per qpos joint around its intrinsic axis.
    g1 = G1(G1_PATH, rotation_type="hinge")
    return {
        "SMPL": smpl,
        "SMPLX": smplx,
        "SKEL": skel,
        "ANNY": anny,
        "MHR": mhr,
        "FLAME": flame,
        "GarmentMeasurements": gm,
        "SOMA": soma,
        "G1": g1,
    }


def main() -> None:
    server = viser.ViserServer()
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/grid", position=(0.0, 0.0, 0.0), plane="xz")
    server.gui.configure_theme(control_layout="fixed", control_width="large")

    models = load_models()
    print(f"Loaded {len(models)} models: {list(models.keys())}", flush=True)

    n = len(models)
    num_rows = (n + GRID_COLS - 1) // GRID_COLS
    states: dict[str, ModelState] = {}
    for i, (name, model) in enumerate(models.items()):
        row, col = divmod(i, GRID_COLS)
        row_count = min(GRID_COLS, n - row * GRID_COLS)
        params = model.get_rest_pose()
        if name == "ANNY":
            params["global_rotation"][0, 0] = ANNY_DISPLAY_ROTATION_X
        verts = model.forward_vertices(**params)
        states[name] = ModelState(
            model=model,
            params=params,
            faces=triangulate(np.asarray(model.faces)),
            color=MODEL_COLORS[name],
            x_offset=(col - 0.5 * (row_count - 1)) * GRID_SPACING_X,
            y_offset=-float(verts[..., 1].min()),
            z_offset=(row - 0.5 * (num_rows - 1)) * GRID_SPACING_Z,
        )

    tabs = server.gui.add_tab_group()
    for name, state in states.items():
        TAB_BUILDERS[name](server, tabs, state)

    for name, state in states.items():
        verts = state.model.forward_vertices(**state.params)
        label_y = float(verts[..., 1].max()) + state.y_offset + 0.1
        server.scene.add_label(f"/labels/{name}", text=name, position=(state.x_offset, label_y, state.z_offset))

    print("\nServer running at http://localhost:8080", flush=True)
    while True:
        time.sleep(0.02)
        for name, state in states.items():
            if state.changed:
                state.changed = False
                update_mesh(server, name, state)


if __name__ == "__main__":
    main()
