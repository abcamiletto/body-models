# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "body-models",
#   "numpy>=2.4.1",
#   "viser>=0.2.32",
# ]
# [tool.uv.sources]
# body-models = { path = "..", editable = true }
# ///
"""Visualize all body models side by side with interactive controls.

Model assets are resolved through the body-models config/cache.

Usage:
    uv run scripts/visualize_models.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import viser

from body_models.anny.numpy import ANNY
from body_models.base import BodyModel
from body_models.brainco.numpy import BrainCoHand
from body_models.constants import Joint
from body_models.flame.numpy import FLAME
from body_models.g1.numpy import G1
from body_models.garment_measurements.numpy import GarmentMeasurements
from body_models.mano.numpy import MANO
from body_models.mhr.numpy import MHR
from body_models.myofullbody.numpy import MyoFullBody
from body_models.skel.numpy import SKEL
from body_models.smpl.numpy import SMPL
from body_models.smplh.numpy import SMPLH
from body_models.smplx.numpy import SMPLX
from body_models.soma.numpy import SOMA

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

# qpos joint indices into model.qpos_joint_names. body_pose stores one scalar
# value per qpos entry (hinge angle or slide displacement); each joint = one slider.
MYOFULLBODY_POSE_JOINTS = [
    ("Lumbar Flex", 0),
    ("Lumbar Bend", 1),
    ("Lumbar Rot", 2),
    ("L Shoulder Plane", 66),
    ("L Shoulder Elev", 67),
    ("L Shoulder Rot", 69),
    ("L Elbow", 70),
    ("R Shoulder Plane", 28),
    ("R Shoulder Elev", 29),
    ("R Shoulder Rot", 31),
    ("R Elbow", 32),
    ("L Hip Flex", 108),
    ("L Hip Add", 109),
    ("L Knee", 113),
    ("L Ankle", 116),
    ("R Hip Flex", 94),
    ("R Hip Add", 95),
    ("R Knee", 99),
    ("R Ankle", 102),
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

BRAINCO_POSE_JOINTS = [
    ("Thumb Metacarpal", 0),
    ("Thumb Proximal", 1),
    ("Index", 2),
    ("Middle", 3),
    ("Ring", 4),
    ("Pinky", 5),
]

# Grid layout: split models across rows on the xz ground plane.
GRID_COLS = 5  # max models per row; with 11 models this gives 5 + 5 + 1
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
    "G1": (200, 200, 220),
    "MyoFullBody": (240, 200, 200),
    "BrainCo": (180, 210, 170),
}

JOINT_MARKER_COLOR = (45, 120, 255)
JOINT_HIGHLIGHT_COLOR = (255, 210, 35)
JOINT_MARKER_RADIUS = 0.025
JOINT_HIGHLIGHT_RADIUS = 0.055
HAND_JOINT_NAMES = ("thumb", "index", "middle", "ring", "pinky")


# ── State ────────────────────────────────────────────────────────────────────
@dataclass
class ModelState:
    model: BodyModel
    params: dict[str, np.ndarray]
    faces: np.ndarray
    color: tuple[int, int, int]
    mesh_handle: viser.MeshHandle | None = None
    muscle_handle: viser.LineSegmentsHandle | None = None
    changed: bool = True


@dataclass
class SliderHandle:
    handle: viser.GuiInputHandle
    initial: float
    key: str
    indices: tuple[int, ...]


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

    return SliderHandle(handle, initial, key, indices)


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
        for slider in handles:
            slider.handle.value = slider.initial


def pose_buttons(server: viser.ViserServer, state: ModelState, handles: list[SliderHandle]) -> None:
    with server.gui.add_folder("Canonical Poses"):
        for label, pose_fn in (
            ("T-pose", state.model.get_tpose),
            ("A-pose", state.model.get_apose),
            ("I-pose", state.model.get_ipose),
        ):
            btn = server.gui.add_button(label)

            @btn.on_click
            def _(_, pose_fn=pose_fn):
                preset = pose_fn()
                for key in ("pose", "body_pose", "hand_pose"):
                    if key in preset and key in state.params:
                        state.params[key] = preset[key]
                for slider in handles:
                    if slider.key in state.params:
                        slider.handle.value = float(state.params[slider.key][slider.indices])
                state.changed = True


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
        pose_buttons(server, state, handles)
        reset_button(server, handles)


def smplx_tab(server, tabs, state) -> None:
    smpl_tab(server, tabs, state, name="SMPLX", with_expression=True)


def smplh_tab(server, tabs, state) -> None:
    smpl_tab(server, tabs, state, name="SMPLH")


def mano_tab(server, tabs, state) -> None:
    handles: list[SliderHandle] = []
    with tabs.add_tab("MANO", viser.Icon.USER):
        with server.gui.add_folder("Shape"):
            handles += betas(server, state, key="shape", count=10)
        with server.gui.add_folder("Hand Pose"):
            handles += joint_xyz(server, state, key="hand_pose", joints=[(f"Joint {i}", i) for i in range(15)])
        reset_button(server, handles)


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
        pose_buttons(server, state, handles)
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
        pose_buttons(server, state, handles)
        reset_button(server, handles)


def mhr_tab(server, tabs, state) -> None:
    handles: list[SliderHandle] = []
    with tabs.add_tab("MHR", viser.Icon.USER):
        with server.gui.add_folder("Shape"):
            handles += betas(server, state, key="shape", count=10)
        with server.gui.add_folder("Expression"):
            handles += betas(server, state, key="expression", count=15, prefix="ψ", lo=-2.0, hi=2.0)
        pose_buttons(server, state, handles)
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
        pose_buttons(server, state, handles)
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
        pose_buttons(server, state, handles)
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
        pose_buttons(server, state, handles)
        reset_button(server, handles)


def myofullbody_tab(server, tabs, state) -> None:
    handles: list[SliderHandle] = []
    with tabs.add_tab("MyoFullBody", viser.Icon.USER):
        with server.gui.add_folder("Pose"):
            for label, qpos_idx in MYOFULLBODY_POSE_JOINTS:
                if qpos_idx >= state.model.num_qpos:
                    continue
                lo, hi = (float(x) for x in cast(Any, state.model).weights.qpos_joint_limits[qpos_idx])
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
                        indices=(0, qpos_idx),
                    )
                )
        pose_buttons(server, state, handles)
        reset_button(server, handles)


def brainco_tab(server, tabs, state) -> None:
    handles: list[SliderHandle] = []
    with tabs.add_tab("BrainCo", viser.Icon.USER):
        with server.gui.add_folder("Hinge Pose"):
            for label, qpos_idx in BRAINCO_POSE_JOINTS:
                lo, hi = (float(x) for x in state.model.qpos_joint_limits[qpos_idx])
                handles.append(
                    add_slider(
                        server,
                        state,
                        label,
                        lo=lo,
                        hi=hi,
                        step=0.02,
                        initial=0.0,
                        key="pose",
                        indices=(0, qpos_idx, 0),
                    )
                )
        reset_button(server, handles)


def _joint_label(joint: Joint) -> str:
    return joint.value.replace("_", " ").title()


def _is_hand_joint(joint: Joint) -> bool:
    return any(name in joint.value for name in HAND_JOINT_NAMES)


def _is_left_joint(joint: Joint) -> bool:
    return joint.value.startswith("left_")


def standard_joints_tab(server: viser.ViserServer, tabs, states: dict[str, ModelState]):
    joint_indices = {
        name: {
            joint: state.model.joint_names.index(native_name)
            for joint, native_name in state.model.common_joints.items()
        }
        for name, state in states.items()
    }
    available_joints = [joint for joint in Joint if any(joint in indices for indices in joint_indices.values())]
    body_joints = [joint for joint in available_joints if not _is_hand_joint(joint)]
    hand_joints = [joint for joint in available_joints if _is_hand_joint(joint)]
    hand_sides = {
        "Left hand": [joint for joint in hand_joints if _is_left_joint(joint)],
        "Right hand": [joint for joint in hand_joints if not _is_left_joint(joint)],
    }
    markers: dict[tuple[str, Joint], Any] = {}
    highlights: dict[tuple[str, Joint], Any] = {}
    visible_joints: set[Joint] = set()
    selected_joint: Joint | None = None

    with tabs.add_tab("Standard Joints", viser.Icon.HAND_CLICK):
        toggle_all = server.gui.add_button("Show all")
        checkboxes = {}
        with server.gui.add_folder("Body", expand_by_default=True):
            for joint in body_joints:
                checkboxes[joint] = server.gui.add_checkbox(_joint_label(joint), initial_value=False)
        with server.gui.add_folder("Hands", expand_by_default=True):
            for label in hand_sides:
                checkboxes[label] = server.gui.add_checkbox(label, initial_value=False)

    def select(joint: Joint | None) -> None:
        nonlocal selected_joint
        selected_joint = joint
        for key, marker in markers.items():
            marker.visible = key[1] in visible_joints
            highlights[key].visible = key[1] in visible_joints and key[1] == selected_joint

    for joint in body_joints:
        checkbox = checkboxes[joint]

        @checkbox.on_update
        def _(event, checkbox_joint=joint) -> None:
            if event.target.value:
                visible_joints.add(checkbox_joint)
            else:
                visible_joints.discard(checkbox_joint)
            select(selected_joint)

    for label, joints in hand_sides.items():
        checkbox = checkboxes[label]

        @checkbox.on_update
        def _(event, side_joints=joints) -> None:
            if event.target.value:
                visible_joints.update(side_joints)
            else:
                visible_joints.difference_update(side_joints)
            select(selected_joint)

    @toggle_all.on_click
    def _(_) -> None:
        show_all = len(visible_joints) < len(available_joints)
        toggle_all.label = "Hide all" if show_all else "Show all"
        for checkbox in checkboxes.values():
            checkbox.value = show_all

    for name, state in states.items():
        for joint in joint_indices[name]:
            key = (name, joint)
            marker_path = f"/{name}/standard_joints/{joint.value}"
            marker = server.scene.add_icosphere(
                marker_path,
                radius=JOINT_MARKER_RADIUS,
                color=JOINT_MARKER_COLOR,
                subdivisions=2,
                visible=False,
            )
            highlight = server.scene.add_icosphere(
                f"/{name}/standard_joint_highlights/{joint.value}",
                radius=JOINT_HIGHLIGHT_RADIUS,
                color=JOINT_HIGHLIGHT_COLOR,
                subdivisions=2,
                opacity=0.75,
                visible=False,
            )

            @marker.on_click
            def _(_, clicked_joint=joint) -> None:
                select(clicked_joint)

            @highlight.on_click
            def _(_, clicked_joint=joint) -> None:
                select(clicked_joint)

            markers[key] = marker
            highlights[key] = highlight

    def update_markers() -> None:
        for name, state in states.items():
            skeleton = state.model.forward_skeleton(**state.params)
            joint_positions = np.asarray(skeleton[0, :, :3, 3], dtype=np.float32)
            for joint, joint_index in joint_indices[name].items():
                position = joint_positions[joint_index]
                markers[(name, joint)].position = position
                highlights[(name, joint)].position = position

    update_markers()
    return update_markers


TAB_BUILDERS = {
    "SMPL": smpl_tab,
    "SMPLH": smplh_tab,
    "MANO": mano_tab,
    "SMPLX": smplx_tab,
    "SKEL": skel_tab,
    "ANNY": anny_tab,
    "MHR": mhr_tab,
    "FLAME": flame_tab,
    "GarmentMeasurements": garment_measurements_tab,
    "SOMA": soma_tab,
    "G1": g1_tab,
    "MyoFullBody": myofullbody_tab,
    "BrainCo": brainco_tab,
}


# ── Mesh & main loop ─────────────────────────────────────────────────────────
def triangulate(faces: np.ndarray) -> np.ndarray:
    if faces.shape[1] == 3:
        return faces
    if faces.shape[1] == 4:
        return np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)
    raise ValueError(f"Unsupported face arity: {faces.shape[1]}")


def _muscle_segment_indices(model: BodyModel) -> np.ndarray | None:
    """Flatten a model's tendons into ``[N_segments, 2]`` index pairs into ``world_sites``."""
    if not model.has_tendons:
        return None
    segments: list[tuple[int, int]] = []
    for tendon in cast(Any, model).tendons:
        sites = tendon["site_indices"]
        segments.extend(zip(sites[:-1], sites[1:]))
    return np.asarray(segments, dtype=np.int64)


def update_mesh(server: viser.ViserServer, name: str, state: ModelState) -> None:
    verts = state.model.forward_vertices(**state.params)[0]
    if state.mesh_handle is None:
        state.mesh_handle = server.scene.add_mesh_simple(
            f"/{name}",
            vertices=verts,
            faces=state.faces,
            color=state.color,
        )
    else:
        state.mesh_handle.vertices = verts

    muscle_segment_indices = cast(Any, state.model)._visualizer_muscle_segment_indices
    if muscle_segment_indices is not None:
        skeleton = state.model.forward_skeleton(**state.params)
        sites = np.asarray(cast(Any, state.model).world_sites(skeleton))[0]
        seg_points = sites[muscle_segment_indices].astype(np.float32)
        if state.muscle_handle is None:
            state.muscle_handle = server.scene.add_line_segments(
                f"/{name}/muscles",
                points=seg_points,
                colors=(220, 50, 50),
                line_width=2.0,
            )
        else:
            state.muscle_handle.points = seg_points


def load_models() -> dict[str, BodyModel]:
    print("Loading SMPL", flush=True)
    smpl = SMPL(gender="neutral")
    print("Loading SMPLH", flush=True)
    smplh = SMPLH(gender="neutral")
    print("Loading MANO", flush=True)
    mano = MANO(side="right")
    print("Loading SMPLX", flush=True)
    smplx = SMPLX(gender="neutral")
    print("Loading SKEL", flush=True)
    skel = SKEL(gender="male")
    print("Loading ANNY", flush=True)
    anny = ANNY()
    print("Loading MHR", flush=True)
    mhr = MHR()
    print("Loading FLAME", flush=True)
    flame = FLAME()
    print("Loading GarmentMeasurements", flush=True)
    gm = GarmentMeasurements()
    print("Loading SOMA", flush=True)
    soma = SOMA()
    print("Loading G1", flush=True)
    # Hinge parametrization: one scalar angle per qpos joint around its intrinsic axis.
    g1 = G1(rotation_type="hinge")
    print("Loading MyoFullBody", flush=True)
    myo = MyoFullBody()
    print("Loading BrainCo", flush=True)
    brainco = BrainCoHand(side="right", rotation_type="hinge")
    models = {
        "SMPL": smpl,
        "SMPLH": smplh,
        "MANO": mano,
        "SMPLX": smplx,
        "SKEL": skel,
        "ANNY": anny,
        "MHR": mhr,
        "FLAME": flame,
        "GarmentMeasurements": gm,
        "SOMA": soma,
        "G1": g1,
        "MyoFullBody": myo,
        "BrainCo": brainco,
    }
    for model in models.values():
        cast(Any, model)._visualizer_muscle_segment_indices = _muscle_segment_indices(model)
    return models


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
        params["global_translation"][0] = (
            (col - 0.5 * (row_count - 1)) * GRID_SPACING_X,
            -float(verts[..., 1].min()),
            (row - 0.5 * (num_rows - 1)) * GRID_SPACING_Z,
        )
        states[name] = ModelState(
            model=model,
            params=params,
            faces=triangulate(np.asarray(model.faces)),
            color=MODEL_COLORS[name],
        )

    tabs = server.gui.add_tab_group()
    for name, state in states.items():
        TAB_BUILDERS[name](server, tabs, state)
    update_joint_markers = standard_joints_tab(server, tabs, states)

    for name, state in states.items():
        verts = state.model.forward_vertices(**state.params)
        label_position = np.asarray(state.params["global_translation"][0]).copy()
        label_position[1] = float(verts[..., 1].max()) + 0.1
        server.scene.add_label(f"/labels/{name}", text=name, position=label_position)

    print("\nServer running", flush=True)
    while True:
        time.sleep(0.02)
        markers_changed = False
        for name, state in states.items():
            if state.changed:
                state.changed = False
                update_mesh(server, name, state)
                markers_changed = True
        if markers_changed:
            update_joint_markers()


if __name__ == "__main__":
    main()
