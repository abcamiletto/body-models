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
from nanomanifold import SO3

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

DISPLAY_GLOBAL_ROTATIONS = {
    "ANNY": (-np.pi / 2, 0.0, 0.0),
}

# ── Slider configurations ────────────────────────────────────────────────────
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
CANONICAL_POSE_MODELS = (
    "SMPL",
    "SMPLH",
    "SMPLX",
    "SKEL",
    "ANNY",
    "MHR",
    "GarmentMeasurements",
    "SOMA",
    "G1",
    "MyoFullBody",
)
POSE_PARAM_KEYS = {"body_pose", "head_pose", "hand_pose", "global_rotation", "global_translation"}


# ── State ────────────────────────────────────────────────────────────────────
@dataclass
class ModelState:
    model: BodyModel
    params: dict[str, np.ndarray]
    faces: np.ndarray
    color: tuple[int, int, int]
    display_global_rotation: np.ndarray | None = None
    skinned: bool = False
    hands: str = "default"
    mesh_handle: viser.MeshHandle | None = None
    skinned_mesh_handle: viser.MeshSkinnedHandle | None = None
    link_mesh_handles: list[viser.MeshHandle] | None = None
    muscle_handle: viser.LineSegmentsHandle | None = None
    mesh_dirty: bool = True
    changed: bool = True


@dataclass
class SliderHandle:
    handle: viser.GuiInputHandle
    initial: float
    key: str
    indices: tuple[int, ...]
    mesh_dirty: bool = False


@dataclass
class ModelControls:
    folder: viser.GuiFolderHandle
    sliders: list[SliderHandle]


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
    mesh_dirty: bool = False,
) -> SliderHandle:
    """Create a slider that writes ``state.params[key][indices] = value`` on update."""
    handle = server.gui.add_slider(label, min=lo, max=hi, step=step, initial_value=initial)

    @handle.on_update
    def _(event):
        state.params[key][indices] = event.target.value
        state.mesh_dirty |= mesh_dirty
        state.changed = True

    return SliderHandle(handle, initial, key, indices, mesh_dirty)


def betas(
    server,
    state,
    *,
    key,
    count,
    prefix="β",
    lo=-3.0,
    hi=3.0,
    step=0.1,
    initial=0.0,
    mesh_dirty=True,
) -> list[SliderHandle]:
    """Indexed sliders writing into ``state.params[key][i]``."""
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
            mesh_dirty=mesh_dirty,
        )
        for i in range(count)
    ]


def joint_xyz(server, state, *, key, joints, lo=-1.5, hi=1.5, step=0.05, max_joints=None) -> list[SliderHandle]:
    """X/Y/Z axis-angle sliders per joint, writing into ``state.params[key][joint, axis]``."""
    handles: list[SliderHandle] = []
    for name, idx in joints:
        if max_joints is not None and idx >= max_joints:
            continue
        for ax, axn in enumerate("XYZ"):
            handles.append(
                add_slider(
                    server, state, f"{name} {axn}", lo=lo, hi=hi, step=step, initial=0.0, key=key, indices=(idx, ax)
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


def mutable_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in params.items()}


def apply_pose(state: ModelState, sliders: list[SliderHandle], pose_name: str) -> None:
    pose_fn = getattr(state.model, f"get_{pose_name}")
    preset = pose_fn(hands=state.hands) if state.model.has_hands else pose_fn()
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
    state.changed = True


def apply_hands(state: ModelState, sliders: list[SliderHandle], hands: str) -> None:
    preset = cast(Any, state.model).get_rest_pose(hands=hands)
    state.hands = hands
    state.params["hand_pose"] = np.asarray(preset["hand_pose"]).copy()
    for slider in sliders:
        if slider.key == "hand_pose":
            slider.handle.value = float(state.params[slider.key][slider.indices])
    state.changed = True


def set_gui_visible(handle: Any, visible: bool) -> None:
    handle.visible = visible
    for child in getattr(handle, "_children", {}).values():
        set_gui_visible(child, visible)


def add_model_controls(server: viser.ViserServer, name: str, state: ModelState) -> ModelControls:
    handles: list[SliderHandle] = []
    with server.gui.add_folder(name, expand_by_default=True) as folder:
        if name in {"SMPL", "SMPLH", "SMPLX"}:
            with server.gui.add_folder("Shape"):
                handles += betas(server, state, key="shape", count=10)
            if name == "SMPLX":
                with server.gui.add_folder("Expression"):
                    handles += betas(server, state, key="expression", count=10, prefix="ψ", lo=-2.0, hi=2.0)
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
                            mesh_dirty=True,
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
                handles += betas(server, state, key="expression", count=15, prefix="ψ", lo=-2.0, hi=2.0)

        elif name == "FLAME":
            with server.gui.add_folder("Shape"):
                handles += betas(server, state, key="shape", count=10)
            with server.gui.add_folder("Expression"):
                handles += betas(server, state, key="expression", count=10, prefix="ψ", lo=-2.0, hi=2.0)
            with server.gui.add_folder("Head Pose"):
                handles += joint_xyz(server, state, key="head_pose", joints=FLAME_POSE_JOINTS, lo=-0.5, hi=0.5)

        elif name == "GarmentMeasurements":
            model = cast(Any, state.model)
            with server.gui.add_folder("Shape"):
                handles += betas(server, state, key="shape", count=model.num_shape_components)
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
            model = cast(Any, state.model)
            identity_default = float(state.params["identity"][0])
            with server.gui.add_folder("Identity"):
                handles += betas(
                    server,
                    state,
                    key="identity",
                    count=min(10, model.identity_dim),
                    prefix="ι",
                    lo=-1.0,
                    hi=1.0,
                    step=0.05,
                    initial=identity_default,
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

        elif name == "G1":
            model = cast(Any, state.model)
            add_hinge_sliders(
                server,
                state,
                folder_name="Hinge Pose",
                joints=G1_POSE_JOINTS,
                limits=model.qpos_joint_limits,
                key="body_pose",
                index=lambda qpos_idx: (qpos_idx, 0),
                handles=handles,
                max_qpos=len(model.qpos_joint_indices),
            )

        elif name == "MyoFullBody":
            model = cast(Any, state.model)
            add_hinge_sliders(
                server,
                state,
                folder_name="Pose",
                joints=MYOFULLBODY_POSE_JOINTS,
                limits=model.weights.qpos_joint_limits,
                key="body_pose",
                index=lambda qpos_idx: (qpos_idx,),
                handles=handles,
                max_qpos=model.num_qpos,
            )

        elif name == "BrainCo":
            model = cast(Any, state.model)
            add_hinge_sliders(
                server,
                state,
                folder_name="Hinge Pose",
                joints=BRAINCO_POSE_JOINTS,
                limits=model.qpos_joint_limits,
                key="hand_pose",
                index=lambda qpos_idx: (qpos_idx, 0),
                handles=handles,
            )

        else:
            raise ValueError(f"Unhandled model controls: {name}")

        reset_button(server, handles)
    return ModelControls(folder, handles)


def add_hinge_sliders(
    server: viser.ViserServer,
    state: ModelState,
    *,
    folder_name: str,
    joints,
    limits,
    key: str,
    index,
    handles: list[SliderHandle],
    max_qpos: int | None = None,
) -> None:
    with server.gui.add_folder(folder_name):
        for label, qpos_idx in joints:
            if max_qpos is not None and qpos_idx >= max_qpos:
                continue
            lo, hi = (float(x) for x in limits[qpos_idx])
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
                    key=key,
                    indices=index(qpos_idx),
                )
            )


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

    with tabs.add_tab("Joints", viser.Icon.HAND_CLICK):
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
            marker_path = f"/joints/{name}/standard/{joint.value}"
            marker = server.scene.add_icosphere(
                marker_path,
                radius=JOINT_MARKER_RADIUS,
                color=JOINT_MARKER_COLOR,
                subdivisions=2,
                visible=False,
            )
            highlight = server.scene.add_icosphere(
                f"/joints/{name}/highlights/{joint.value}",
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
            joint_positions = np.asarray(skeleton[:, :3, 3], dtype=np.float32)
            for joint, joint_index in joint_indices[name].items():
                position = joint_positions[joint_index]
                markers[(name, joint)].position = position
                highlights[(name, joint)].position = position

    update_markers()
    return update_markers


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


def supports_viser_skinning(model: BodyModel) -> bool:
    if model.is_rigid_body:
        return False
    try:
        np.asarray(model.skin_weights)
        return True
    except (AttributeError, NotImplementedError):
        return False


def viser_forward_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    forward_params = dict(params)
    forward_params["global_translation"] = np.zeros_like(forward_params["global_translation"])
    return {key: np.asarray(value)[None] for key, value in forward_params.items()}


def skinned_bind_params(state: ModelState) -> dict[str, np.ndarray]:
    if state.model.has_hands:
        params = cast(Any, state.model).get_rest_pose(hands=state.hands)
    else:
        params = state.model.get_rest_pose()
    params = mutable_params(params)
    for key, value in state.params.items():
        if key not in POSE_PARAM_KEYS and key in params:
            params[key] = np.asarray(value).copy()
    return params


def vec3(value: np.ndarray) -> tuple[float, float, float]:
    x, y, z = np.asarray(value, dtype=np.float32)
    return float(x), float(y), float(z)


def update_link_meshes(server: viser.ViserServer, name: str, state: ModelState) -> None:
    model = cast(Any, state.model)
    if state.link_mesh_handles is None:
        state.link_mesh_handles = []
        for index, link_name in enumerate(model.link_names):
            mesh = model.link_mesh(link_name)
            handle = server.scene.add_mesh_simple(
                f"/meshes/{name}/links/{index:02d}",
                vertices=np.asarray(mesh["vertices"], dtype=np.float32),
                faces=np.asarray(mesh["faces"]),
                color=state.color,
            )
            state.link_mesh_handles.append(handle)

    links = np.asarray(model.forward_links(**state.params))
    link_wxyzs = SO3.conversions.from_rotmat_to_quat(links[:, :3, :3], convention="wxyz", xp=np)
    link_positions = links[:, :3, 3]
    for handle, wxyz, position in zip(state.link_mesh_handles, link_wxyzs, link_positions):
        handle.wxyz = wxyz
        handle.position = vec3(position)


def update_mesh(server: viser.ViserServer, name: str, state: ModelState) -> None:
    if state.skinned and (state.skinned_mesh_handle is None or state.mesh_dirty):
        if state.skinned_mesh_handle is not None:
            state.skinned_mesh_handle.remove()

        mesh = state.model.to_viser_skinned_mesh(**viser_forward_params(skinned_bind_params(state)))
        state.skinned_mesh_handle = server.scene.add_mesh_skinned(
            f"/meshes/{name}",
            vertices=mesh["vertices"],
            faces=mesh["faces"],
            bone_wxyzs=mesh["bone_wxyzs"],
            bone_positions=mesh["bone_positions"],
            skin_weights=mesh["skin_weights"],
            color=state.color,
            position=vec3(state.params["global_translation"]),
        )
        state.mesh_dirty = False
    if state.skinned:
        handle = state.skinned_mesh_handle
        assert handle is not None
        bones = state.model.to_viser_bones(**viser_forward_params(state.params))
        handle.position = vec3(state.params["global_translation"])
        for bone, wxyz, bone_position in zip(
            handle.bones,
            bones["bone_wxyzs"],
            bones["bone_positions"],
        ):
            bone.wxyz = wxyz
            bone.position = vec3(bone_position)
    elif state.model.is_rigid_body:
        update_link_meshes(server, name, state)
    else:
        vertices = state.model.forward_vertices(**state.params)
        if state.mesh_handle is None:
            state.mesh_handle = server.scene.add_mesh_simple(
                f"/meshes/{name}",
                vertices=vertices,
                faces=state.faces,
                color=state.color,
            )
        else:
            state.mesh_handle.vertices = vertices

    muscle_segment_indices = cast(Any, state.model)._visualizer_muscle_segment_indices
    if muscle_segment_indices is not None:
        skeleton = state.model.forward_skeleton(**state.params)
        sites = np.asarray(cast(Any, state.model).world_sites(skeleton))
        seg_points = sites[muscle_segment_indices].astype(np.float32)
        if state.muscle_handle is None:
            state.muscle_handle = server.scene.add_line_segments(
                f"/muscles/{name}",
                points=seg_points,
                colors=(220, 50, 50),
                line_width=2.0,
            )
        else:
            state.muscle_handle.points = seg_points


def load_models() -> dict[str, BodyModel]:
    model_specs = (
        ("SMPL", lambda: SMPL(gender="neutral")),
        ("SMPLH", lambda: SMPLH(gender="neutral")),
        ("MANO", lambda: MANO(side="right")),
        ("SMPLX", lambda: SMPLX(gender="neutral")),
        ("SKEL", lambda: SKEL(gender="male")),
        ("ANNY", ANNY),
        ("MHR", MHR),
        ("FLAME", FLAME),
        ("GarmentMeasurements", GarmentMeasurements),
        ("SOMA", SOMA),
        ("G1", lambda: G1(rotation_type="hinge")),
        ("MyoFullBody", MyoFullBody),
        ("BrainCo", lambda: BrainCoHand(side="right", rotation_type="hinge")),
    )
    models = {}
    for name, make_model in model_specs:
        print(f"Loading {name}", flush=True)
        models[name] = make_model()
    for model in models.values():
        cast(Any, model)._visualizer_muscle_segment_indices = _muscle_segment_indices(model)
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
            faces=triangulate(np.asarray(model.faces)),
            color=MODEL_COLORS[name],
            display_global_rotation=display_global_rotation,
            skinned=supports_viser_skinning(model),
        )
    return states


def add_labels(server: viser.ViserServer, states: dict[str, ModelState]) -> None:
    for name, state in states.items():
        verts = state.model.forward_vertices(**state.params)
        label_position = np.asarray(state.params["global_translation"]).copy()
        label_position[1] = float(verts[..., 1].max()) + 0.1
        server.scene.add_label(f"/labels/{name}", text=name, position=label_position)


def main() -> None:
    server = viser.ViserServer()
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/grid", position=(0.0, 0.0, 0.0), plane="xz")
    server.gui.configure_theme(control_layout="fixed", control_width="large")

    models = load_models()
    print(f"Loaded {len(models)} models: {list(models.keys())}", flush=True)

    states = init_states(models)

    tabs = server.gui.add_tab_group()
    selected_model = next(iter(states))
    with tabs.add_tab("Models", viser.Icon.USER):
        model_dropdown = server.gui.add_dropdown(
            "Model",
            options=tuple(states.keys()),
            initial_value=selected_model,
        )
        controls = {name: add_model_controls(server, name, state) for name, state in states.items()}

    with tabs.add_tab("Poses"):
        with server.gui.add_folder("Body"):
            for label, pose_name in (("T-pose", "tpose"), ("A-pose", "apose")):
                button = server.gui.add_button(label)

                @button.on_click
                def _(_, pose_name=pose_name) -> None:
                    for name in CANONICAL_POSE_MODELS:
                        state = states[name]
                        apply_pose(state, controls[name].sliders, pose_name)

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

    update_joint_markers = standard_joints_tab(server, tabs, states)
    add_labels(server, states)

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
