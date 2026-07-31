"""Generate rigid SMPL robot assets and reviewable GLB snapshots."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from importlib.resources import files
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from trimesh.visual.material import PBRMaterial

from body_models import smpl_humanoid
from body_models.smpl import SMPL
from body_models.smpl_humanoid import SmplMannequin, generate
from body_models.smpl_humanoid import _constants as constants
from body_models.smplx import SMPLX

SHAPES = {
    "neutral": np.zeros(10, dtype=np.float32),
    "long_limb": np.array([1.2, -2.5, 0.8, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "compact": np.array([-1.4, 2.8, -0.8, 0.5, 0, 0, 0, 0, 0, 0], dtype=np.float32),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smpl_robot"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    smpl = SMPL(model_path=args.model)
    smplx = SMPLX(
        model_path=args.smplx_model,
        gender=None if args.smplx_model else "neutral",
        flat_hand_mean=True,
    )
    neutral_identity = smpl.prepare_identity(np.zeros(10, dtype=np.float32))
    raw_neutral_offsets = np.asarray(neutral_identity["local_joint_offsets"])
    reference_offsets = generate._smplx_reference_offsets(smplx)
    report = {
        "rig_type": "54 rigid joints; no skinning",
        "units": "meters",
        "identities": {},
    }
    for name, shape in SHAPES.items():
        xml_path = smpl_humanoid.generate_smpl_robot(
            args.output / f"{name}.xml",
            shape=shape,
            source_model=smpl,
            smplx_model=smplx,
        )
        robot = SmplMannequin(model_path=xml_path)
        source_identity = smpl.prepare_identity(shape)
        robot_joints = robot.forward_skeleton(**robot.get_tpose())[:, :3, 3]
        robot_size = robot.forward_meshes(**robot.get_tpose())[0].extents
        expected_offsets = generate._shape_scaled_offsets(
            reference_offsets,
            smpl_neutral_offsets=raw_neutral_offsets,
            smpl_measured_offsets=np.asarray(source_identity["local_joint_offsets"]),
        )
        expected_joints = generate._joints_from_offsets(
            np.zeros(3),
            expected_offsets,
            parents=constants.ROBOT_PARENTS,
        )
        body_parents = robot.parents[:24]
        source_bone_lengths = _bone_lengths(expected_joints, body_parents)
        robot_bone_lengths = _bone_lengths(robot_joints[:24], body_parents)
        report["identities"][name] = {
            "shape": shape.tolist(),
            "robot_size_xyz": robot_size.tolist(),
            "max_bone_length_error_m": float(np.abs(robot_bone_lengths - source_bone_lengths).max()),
            "max_bilateral_mesh_error_m": _max_bilateral_mesh_error(robot),
            "head_link_size_xyz": _joint_link_size(robot, 15).tolist(),
            "left_hand_link_size_xyz": _joint_link_size(robot, 22).tolist(),
            "joint_count": robot.num_joints,
            "pose_dofs": robot.num_actuated,
            "skin_weights": 0,
            "rigid_mesh_parts": len(robot.link_names),
            "render_triangle_count": len(robot.faces),
            "mesh_format": "OBJ",
        }
        _export_scene(robot, xml_path, robot.get_tpose(), args.output / f"{name}_tpose.glb")
        _export_scene(robot, xml_path, robot.get_apose(), args.output / f"{name}_apose.glb")

        if name == "neutral":
            smpl_pose = np.zeros((23, 3), dtype=np.float32)
            smpl_pose[0] = [0.10, -0.10, 0.35]
            smpl_pose[1] = [-0.10, 0.05, -0.55]
            smpl_pose[3] = [0.65, 0.0, 0.0]
            smpl_pose[4] = [1.05, 0.0, 0.0]
            smpl_pose[15] = [-0.35, 0.20, 0.65]
            smpl_pose[16] = [-0.20, -0.15, -0.85]
            smpl_pose[17] = [-0.85, -0.15, -0.25]
            smpl_pose[18] = [-1.10, 0.10, 0.30]
            left_hand_pose = np.zeros((15, 3), dtype=np.float32)
            right_hand_pose = np.zeros((15, 3), dtype=np.float32)
            left_hand_pose[:, 1] = np.linspace(0.15, 0.65, 15)
            right_hand_pose[:, 1] = np.linspace(-0.15, -0.65, 15)
            motion = robot.parameters_from_smpl(
                smpl_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
            )
            _export_scene(robot, xml_path, motion, args.output / "neutral_action.glb")

    neutral_robot = SmplMannequin(model_path=args.output / "neutral.xml")
    fk_report = _fk_comparison(neutral_robot, smplx)
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "fk_comparison.json").write_text(
        json.dumps(fk_report, indent=2) + "\n",
        encoding="utf-8",
    )
    license_text = (files("body_models.smpl_humanoid") / "assets/CHARACTER_ASSET.md").read_text(encoding="utf-8")
    (args.output / "ATTRIBUTION.md").write_text(license_text, encoding="utf-8")
    print(f"Wrote rigid robot artifacts to {args.output}")


def _fk_comparison(robot: SmplMannequin, smplx: SMPLX) -> dict[str, object]:
    body = np.zeros((1, 21, 3), dtype=np.float32)
    hand = np.zeros((1, 30, 3), dtype=np.float32)
    robot_fk, smplx_fk, names = _matched_fk(robot, smplx, body, hand)
    return {
        "rest": _fk_error_summary(robot_fk, smplx_fk, names),
        "dof_sweep": _dof_sweep(robot, smplx),
    }


def _dof_sweep(
    robot: SmplMannequin,
    smplx: SMPLX,
    angle: float = 0.15,
) -> dict[str, object]:
    source_joint_names = [*smplx.joint_names[1:22], *smplx.joint_names[25:]]
    num_dofs = len(source_joint_names) * 3
    poses = np.zeros((num_dofs * 2, len(source_joint_names), 3), dtype=np.float32)
    for joint_index in range(len(source_joint_names)):
        for axis in range(3):
            dof_index = joint_index * 3 + axis
            poses[dof_index * 2, joint_index, axis] = angle
            poses[dof_index * 2 + 1, joint_index, axis] = -angle

    labels = [f"{joint}.{axis}" for joint in source_joint_names for axis in "XYZ"]
    robot_fk, smplx_fk, _ = _matched_fk(
        robot,
        smplx,
        poses[:, :21],
        poses[:, 21:],
    )
    rotation_errors, _ = _fk_errors(robot_fk, smplx_fk)
    robot_positions = _root_relative_positions(robot_fk)
    smplx_positions = _root_relative_positions(smplx_fk)
    robot_response = robot_positions[::2] - robot_positions[1::2]
    smplx_response = smplx_positions[::2] - smplx_positions[1::2]
    response_dot = np.sum(robot_response * smplx_response, axis=(1, 2))
    response_norm = np.linalg.norm(robot_response, axis=(1, 2)) * np.linalg.norm(
        smplx_response,
        axis=(1, 2),
    )
    observable = response_norm > 1e-8
    direction_cosines = np.ones(num_dofs, dtype=np.float32)
    direction_cosines[observable] = response_dot[observable] / response_norm[observable]
    rotation_errors = rotation_errors.reshape(num_dofs, 2, -1).max(axis=(1, 2))
    passed = (rotation_errors < 0.1) & (~observable | (direction_cosines > 0.5))
    worst_direction = np.flatnonzero(observable)[np.argmin(direction_cosines[observable])]
    return {
        "angle_degrees": float(np.degrees(angle)),
        "native_smplx_dofs": num_dofs,
        "signed_samples": num_dofs * 2,
        "passed_dofs": int(passed.sum()),
        "failed_dofs": [label for label, ok in zip(labels, passed, strict=True) if not ok],
        "max_rotation_error_degrees": float(rotation_errors.max()),
        "min_observable_position_response_cosine": float(direction_cosines[observable].min()),
        "worst_position_response_dof": labels[worst_direction],
        "orientation_only_dofs": int((~observable).sum()),
        "mannequin_only_dofs": [f"{side}_Hand.{axis}" for side in ("L", "R") for axis in "XYZ"],
    }


def _matched_fk(
    robot: SmplMannequin,
    smplx: SMPLX,
    body_pose: np.ndarray,
    hand_pose: np.ndarray,
    root_pose: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    smplx_pose = smplx.get_tpose(batch_dims=body_pose.shape[:-2])
    smplx_pose["body_pose"] = body_pose
    smplx_pose["hand_pose"] = hand_pose
    robot_pose = robot.parameters_from_smpl(
        body_pose,
        global_rotation=root_pose,
        left_hand_pose=hand_pose[:, :15],
        right_hand_pose=hand_pose[:, 15:],
    )
    if root_pose is not None:
        smplx_pose["pelvis_rotation"] = root_pose

    robot_fk = np.asarray(robot.forward_skeleton(**robot_pose))
    smplx_fk = np.asarray(smplx.forward_skeleton(**smplx_pose))
    name_overrides = {
        "Torso": "Spine1",
        "Spine": "Spine2",
        "Chest": "Spine3",
        "L_Toe": "L_Foot",
        "R_Toe": "R_Foot",
        "L_Thorax": "L_Collar",
        "R_Thorax": "R_Collar",
    }
    pairs = [
        (robot_index, smplx.joint_names.index(name_overrides.get(name, name)), name)
        for robot_index, name in enumerate(robot.joint_names)
        if name not in {"L_Hand", "R_Hand"} and name_overrides.get(name, name) in smplx.joint_names
    ]
    robot_fk = np.take(robot_fk, [pair[0] for pair in pairs], axis=-3)
    smplx_fk = np.take(smplx_fk, [pair[1] for pair in pairs], axis=-3)
    return robot_fk, smplx_fk, [pair[2] for pair in pairs]


def _fk_error_summary(
    robot_fk: np.ndarray,
    smplx_fk: np.ndarray,
    names: list[str],
) -> dict[str, object]:
    rotation_errors, position_errors = _fk_errors(robot_fk, smplx_fk)
    worst_frame, worst_joint = np.unravel_index(
        np.argmax(position_errors),
        position_errors.shape,
    )
    return {
        "max_rotation_error_degrees": float(rotation_errors.max()),
        "mean_joint_position_error_mm": float(position_errors.mean() * 1000.0),
        "max_joint_position_error_mm": float(position_errors.max() * 1000.0),
        "worst_frame": int(worst_frame),
        "worst_joint": names[worst_joint],
    }


def _fk_errors(
    robot_fk: np.ndarray,
    smplx_fk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    relative_rotations = np.swapaxes(robot_fk[..., :3, :3], -1, -2) @ smplx_fk[..., :3, :3]
    rotation_cosines = np.clip(
        (np.trace(relative_rotations, axis1=-2, axis2=-1) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    rotation_errors = np.degrees(np.arccos(rotation_cosines))
    position_errors = np.linalg.norm(
        _root_relative_positions(robot_fk) - _root_relative_positions(smplx_fk),
        axis=-1,
    )
    return rotation_errors, position_errors


def _root_relative_positions(transforms: np.ndarray) -> np.ndarray:
    positions = transforms[..., :3, 3]
    return positions - positions[..., :1, :]


def _bone_lengths(joints: np.ndarray, parents: list[int]) -> np.ndarray:
    return np.asarray(
        [np.linalg.norm(joints[index] - joints[parent]) for index, parent in enumerate(parents) if parent >= 0]
    )


def _joint_link_size(robot: SmplMannequin, joint_index: int) -> np.ndarray:
    vertices = []
    for link_index, link_joint_index in enumerate(robot.link_joint_indices):
        if link_joint_index != joint_index:
            continue
        start = robot.link_vertex_starts[link_index]
        count = robot.link_vertex_counts[link_index]
        local = robot._weights.vertices[start : start + count]
        local = local @ robot.link_geom_rotations[link_index].T + robot.link_geom_positions[link_index]
        vertices.append(local)
    return np.ptp(np.concatenate(vertices), axis=0)


def _max_bilateral_mesh_error(robot: SmplMannequin) -> float:
    transforms = np.asarray(robot.forward_links(**robot.get_tpose()))
    if transforms.ndim == 4:
        transforms = transforms[0]
    vertices_by_joint: dict[int, list[np.ndarray]] = {}
    for link_index, joint_index in enumerate(robot.link_joint_indices):
        start = robot.link_vertex_starts[link_index]
        count = robot.link_vertex_counts[link_index]
        vertices = robot._weights.vertices[start : start + count]
        vertices = vertices @ robot.link_geom_rotations[link_index].T
        vertices += robot.link_geom_positions[link_index]
        vertices = vertices @ transforms[link_index, :3, :3].T
        vertices += transforms[link_index, :3, 3]
        vertices_by_joint.setdefault(joint_index, []).append(vertices)

    maximum = 0.0
    for left, right in ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)):
        left_vertices = np.concatenate(vertices_by_joint[left])
        right_vertices = np.concatenate(vertices_by_joint[right]).copy()
        right_vertices[:, 0] *= -1.0
        distances, _ = cKDTree(right_vertices).query(left_vertices)
        maximum = max(maximum, float(distances.max()))
    return maximum


def _export_scene(robot: SmplMannequin, xml_path: Path, params: dict[str, np.ndarray], output_path: Path) -> None:
    transforms = np.asarray(robot.forward_links(**params))
    if transforms.ndim == 4:
        transforms = transforms[0]
    colors = [
        np.fromstring(geom.get("rgba", "0.8 0.8 0.8 1"), sep=" ")
        for body_name in robot.joint_names
        for geom in ET.parse(xml_path).getroot().find(f".//body[@name='{body_name}']").findall("geom")
    ]
    scene = trimesh.Scene()
    for link_index, (start, count, face_start, face_count) in enumerate(
        zip(
            robot.link_vertex_starts,
            robot.link_vertex_counts,
            robot.link_face_starts,
            robot.link_face_counts,
            strict=True,
        )
    ):
        vertices = robot._weights.vertices[start : start + count]
        faces = robot.faces[face_start : face_start + face_count] - start
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        mesh.remove_unreferenced_vertices()
        color = colors[link_index]
        mesh.visual.material = PBRMaterial(
            baseColorFactor=color,
            metallicFactor=0.18,
            roughnessFactor=0.34,
        )
        scene.add_geometry(
            mesh,
            node_name=robot.link_names[link_index],
            geom_name=robot.link_names[link_index],
            transform=transforms[link_index],
        )
    scene.export(output_path)


if __name__ == "__main__":
    main()
