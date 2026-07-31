"""Play Hugging Face SMPL-X motions on SMPL-X and the rigid mannequin."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser
from huggingface_hub import hf_hub_download
from nanomanifold import SO3
from visualize_viser import add_rigid_meshes, read_link_colors

from body_models.smpl_humanoid import SmplxMannequin
from body_models.smpl_humanoid._io import get_model_path
from body_models.smplx import SMPLX

MODEL_SEPARATION = 1.0
MAX_PLAYBACK_FPS = 30.0
Z_UP_TO_Y_UP = np.array(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 0.0),
    ),
    dtype=np.float32,
)


@dataclass(frozen=True)
class MotionSource:
    label: str
    repo: str
    path: str
    description: str
    z_up: bool = False


@dataclass(frozen=True)
class Motion:
    label: str
    description: str
    fps: float
    body_pose: np.ndarray
    hand_pose: np.ndarray
    head_pose: np.ndarray
    pelvis_rotation: np.ndarray
    global_rotation: np.ndarray
    translation: np.ndarray
    shape: np.ndarray

    @property
    def frames(self) -> int:
        return len(self.body_pose)

    @property
    def duration(self) -> float:
        return max((self.frames - 1) / self.fps, 0.0)


SOURCES = (
    MotionSource(
        "AMASS · walking",
        "ai-habitat/habitat_humanoids",
        "walk_motion/CMU_10_04_stageii.npz",
        "CMU subject 10 walking sequence, released by Habitat Humanoids as an SMPL-X AMASS motion.",
        z_up=True,
    ),
    MotionSource(
        "HumanAct12 · squat jump",
        "ZeyuLing/MotionHub",
        "HumanML3D_HumanACT12/smplh_52/000001.npz",
        "Squat down low, then jump up quickly.",
    ),
    MotionSource(
        "HumanAct12 · balance and gesture",
        "ZeyuLing/MotionHub",
        "HumanML3D_HumanACT12/smplh_52/000011.npz",
        "Sit, gesture with both hands, then balance on the right leg.",
    ),
    MotionSource(
        "HumanAct12 · arm raises",
        "ZeyuLing/MotionHub",
        "HumanML3D_HumanACT12/smplh_52/000053.npz",
        "Repeatedly raise and lower both arms.",
    ),
)


def download_motions(cache_dir: Path | None = None) -> list[Motion]:
    """Download the small named samples and normalize them to Y-up SMPL-X."""
    motions = []
    for source in SOURCES:
        path = hf_hub_download(
            source.repo,
            source.path,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        motions.append(load_motion(Path(path), source))
    return motions


def load_motion(path: Path, source: MotionSource) -> Motion:
    """Load either a standard AMASS SMPL-X file or a MotionHub SMPL-H file."""
    data = np.load(path, allow_pickle=False)
    fps = float(_array(data, "mocap_framerate", "mocap_frame_rate"))
    body_pose = _array(data, "body_pose", "pose_body").reshape(-1, 21, 3)
    pelvis_rotation = _array(data, "global_orient", "root_orient").reshape(-1, 3)
    translation = _array(data, "transl", "trans").reshape(-1, 3).copy()

    if "pose_hand" in data:
        hand_pose = np.asarray(data["pose_hand"]).reshape(-1, 30, 3)
    else:
        left = np.asarray(data["left_hand_pose"]).reshape(-1, 15, 3)
        right = np.asarray(data["right_hand_pose"]).reshape(-1, 15, 3)
        hand_pose = np.concatenate((left, right), axis=1)

    head_pose = np.zeros((len(body_pose), 3, 3), dtype=np.float32)
    if "pose_jaw" in data:
        head_pose[:, 0] = np.asarray(data["pose_jaw"]).reshape(-1, 3)
    if "pose_eye" in data:
        head_pose[:, 1:] = np.asarray(data["pose_eye"]).reshape(-1, 2, 3)

    global_rotation = np.zeros((len(body_pose), 3), dtype=np.float32)
    if source.z_up:
        translation = translation @ Z_UP_TO_Y_UP.T
        rotation = SO3.conversions.from_rotmat_to_axis_angle(Z_UP_TO_Y_UP, xp=np)
        global_rotation[:] = rotation

    horizontal = translation[:, (0, 2)]
    translation[:, (0, 2)] -= (horizontal.min(axis=0) + horizontal.max(axis=0)) / 2.0

    stride = max(1, round(fps / MAX_PLAYBACK_FPS))
    frame_slice = slice(None, None, stride)
    return Motion(
        label=source.label,
        description=source.description,
        fps=fps / stride,
        body_pose=np.asarray(body_pose[frame_slice], dtype=np.float32),
        hand_pose=np.asarray(hand_pose[frame_slice], dtype=np.float32),
        head_pose=np.asarray(head_pose[frame_slice], dtype=np.float32),
        pelvis_rotation=np.asarray(pelvis_rotation[frame_slice], dtype=np.float32),
        global_rotation=np.asarray(global_rotation[frame_slice], dtype=np.float32),
        translation=np.asarray(translation[frame_slice], dtype=np.float32),
        shape=np.asarray(data["betas"][:10], dtype=np.float32),
    )


def _array(data: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    raise ValueError(f"Motion is missing all supported keys: {', '.join(names)}")


def build_viewer(
    xml_path: Path,
    motions: list[Motion],
    *,
    smplx_model: Path | None,
    host: str,
    port: int,
) -> tuple[viser.ViserServer, Callable[[], None]]:
    """Create the synchronized motion viewer and return its playback tick."""
    smplx = SMPLX(model_path=smplx_model, gender=None if smplx_model else "neutral", flat_hand_mean=True)
    mannequin = SmplxMannequin(model_path=xml_path, smplx_model=smplx)
    expressions = np.zeros(10, dtype=np.float32)
    smplx_identities = {
        motion.label: smplx.prepare_identity(motion.shape, expression=expressions) for motion in motions
    }
    mannequin_identities = {motion.label: mannequin.prepare_identity(motion.shape) for motion in motions}
    smplx_offsets = {
        motion.label: root_alignment(
            mannequin,
            smplx,
            mannequin_identities[motion.label],
            smplx_identities[motion.label],
            motion,
        )
        for motion in motions
    }

    server = viser.ViserServer(host=host, port=port, label="SMPL-X motion comparison")
    server.gui.add_markdown("**Left:** rigid mannequin · **Right:** skinned SMPL-X")
    source_info = server.gui.add_markdown("")
    server.scene.set_up_direction("+y")
    mannequin_frame = server.scene.add_frame(
        "/mannequin",
        show_axes=False,
        position=(-MODEL_SEPARATION, 0.0, 0.0),
    )
    smplx_frame = server.scene.add_frame(
        "/smplx",
        show_axes=False,
        position=(MODEL_SEPARATION, 0.0, 0.0),
    )
    server.scene.add_grid(
        "/ground",
        width=8.0,
        height=8.0,
        plane="xz",
        cell_size=0.1,
        section_size=0.5,
        cell_color=(205, 195, 180),
        section_color=(150, 135, 115),
        plane_color=(242, 235, 220),
        plane_opacity=0.35,
    )
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)

    mannequin_meshes = add_rigid_meshes(
        server,
        mannequin,
        read_link_colors(xml_path, mannequin.link_names),
    )
    smplx_mesh = server.scene.add_mesh_simple(
        "/smplx/skinned_mesh",
        vertices=np.asarray(smplx_identities[motions[0].label]["rest_vertices"]),
        faces=smplx.faces,
        color=(164, 197, 207),
        material="standard",
        flat_shading=False,
        side="double",
        cast_shadow=True,
        receive_shadow=True,
    )

    with server.gui.add_folder("Playback", expand_by_default=True):
        motion_control = server.gui.add_dropdown(
            "Motion",
            options=[motion.label for motion in motions],
        )
        playing = server.gui.add_checkbox("Playing", initial_value=True)
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        root_motion = server.gui.add_checkbox("Root motion", initial_value=True)
        speed = server.gui.add_slider("Speed", min=0.25, max=2.0, step=0.05, initial_value=1.0)
        progress = server.gui.add_slider("Timeline", min=0.0, max=1.0, step=0.001, initial_value=0.0)
        time_info = server.gui.add_markdown("")

    with server.gui.add_folder("Display"):
        show_mannequin = server.gui.add_checkbox("Show rigid mannequin", initial_value=True)
        show_smplx = server.gui.add_checkbox("Show skinned SMPL-X", initial_value=True)

    motion_by_label = {motion.label: motion for motion in motions}
    syncing = False
    displayed_identity = None
    previous_time = time.perf_counter()

    def current_motion() -> Motion:
        return motion_by_label[motion_control.value]

    def render() -> None:
        nonlocal displayed_identity
        motion = current_motion()
        frame_index = min(round(float(progress.value) * (motion.frames - 1)), motion.frames - 1)
        translation = motion.translation[frame_index].copy()
        if not root_motion.value:
            translation[(0, 2)] = 0.0

        body_pose = motion.body_pose[frame_index]
        hand_pose = motion.hand_pose[frame_index]
        head_pose = motion.head_pose[frame_index]
        pelvis_rotation = motion.pelvis_rotation[frame_index]
        global_rotation = motion.global_rotation[frame_index]
        smplx_identity = smplx_identities[motion.label]
        mannequin_identity = mannequin_identities[motion.label]
        vertices = np.asarray(
            smplx.forward_vertices(
                body_pose,
                hand_pose,
                head_pose,
                pelvis_rotation=pelvis_rotation,
                global_rotation=global_rotation,
                global_translation=translation,
                identity=smplx_identity,
            )
        )
        transforms = np.asarray(
            mannequin.forward_skeleton(
                body_pose,
                hand_pose,
                head_pose,
                pelvis_rotation=pelvis_rotation,
                global_rotation=global_rotation,
                global_translation=translation,
                identity=mannequin_identity,
            )
        )

        if displayed_identity != motion.label:
            local_vertices = mannequin_identity["link_local_vertices"]
            for handle, start, count in zip(
                mannequin_meshes,
                mannequin.link_vertex_starts,
                mannequin.link_vertex_counts,
                strict=True,
            ):
                handle.vertices = local_vertices[start : start + count]
            displayed_identity = motion.label

        with server.atomic():
            smplx_frame.position = np.array((MODEL_SEPARATION, 0.0, 0.0)) + smplx_offsets[motion.label]
            for handle, joint_index in zip(mannequin_meshes, mannequin.link_joint_indices, strict=True):
                transform = transforms[joint_index]
                handle.position = transform[:3, 3]
                handle.wxyz = SO3.conversions.from_rotmat_to_quat(
                    transform[:3, :3],
                    convention="wxyz",
                    xp=np,
                )
            smplx_mesh.vertices = vertices
            time_info.content = (
                f"Frame **{frame_index + 1}/{motion.frames}** · "
                f"**{frame_index / motion.fps:.2f}/{motion.duration:.2f} s** · "
                f"**{motion.fps:g} fps**"
            )

    def reset_motion() -> None:
        nonlocal syncing, previous_time
        motion = current_motion()
        syncing = True
        progress.value = 0.0
        syncing = False
        previous_time = time.perf_counter()
        source_info.content = f"**{motion.label}**  \n{motion.description}"
        render()

    @motion_control.on_update
    def _(_event) -> None:
        reset_motion()

    @progress.on_update
    def _(_event) -> None:
        if not syncing:
            render()

    @root_motion.on_update
    def _(_event) -> None:
        render()

    @show_mannequin.on_update
    def _(_event) -> None:
        mannequin_frame.visible = show_mannequin.value

    @show_smplx.on_update
    def _(_event) -> None:
        smplx_frame.visible = show_smplx.value

    def tick() -> None:
        nonlocal syncing, previous_time
        now = time.perf_counter()
        elapsed = now - previous_time
        previous_time = now
        if not playing.value:
            return

        motion = current_motion()
        next_time = float(progress.value) * motion.duration + elapsed * float(speed.value)
        if next_time >= motion.duration:
            if loop.value:
                next_time %= motion.duration
            else:
                next_time = motion.duration
                playing.value = False
        syncing = True
        progress.value = next_time / max(motion.duration, 1e-8)
        syncing = False
        render()

    reset_motion()
    server.initial_camera.position = (0.0, 1.0, 8.0)
    server.initial_camera.look_at = (0.0, 0.9, 0.0)
    server.initial_camera.up = (0.0, 1.0, 0.0)
    server.initial_camera.fov = np.deg2rad(40.0)
    return server, tick


def root_alignment(
    mannequin: SmplxMannequin,
    smplx: SMPLX,
    mannequin_identity,
    smplx_identity,
    motion: Motion,
) -> np.ndarray:
    """Return the static offset that makes the two model roots coincide."""
    frame = 0
    body_pose = motion.body_pose[frame]
    hand_pose = motion.hand_pose[frame]
    head_pose = motion.head_pose[frame]
    pelvis_rotation = motion.pelvis_rotation[frame]
    global_rotation = motion.global_rotation[frame]
    translation = motion.translation[frame]
    smplx_root = np.asarray(
        smplx.forward_skeleton(
            body_pose,
            hand_pose,
            head_pose,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=translation,
            identity=smplx_identity,
        )
    )[0, :3, 3]
    mannequin_root = np.asarray(
        mannequin.forward_skeleton(
            body_pose,
            hand_pose,
            head_pose,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=translation,
            identity=mannequin_identity,
        )
    )[0, :3, 3]
    return mannequin_root - smplx_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        default="mannequin",
        help="Named mannequin source or generated MJCF path.",
    )
    parser.add_argument(
        "--smplx-model",
        type=Path,
        help="Neutral SMPL-X .npz/.pkl. Defaults to the configured smplx-neutral model.",
    )
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face cache directory.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_path = get_model_path(args.model).resolve()
    motions = download_motions(args.cache_dir)
    server, tick = build_viewer(
        xml_path,
        motions,
        smplx_model=args.smplx_model,
        host=args.host,
        port=args.port,
    )
    try:
        while True:
            tick()
            time.sleep(1.0 / MAX_PLAYBACK_FPS)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
