"""Fit one body model mesh to another and save barycentric topology maps.

The target mesh stays fixed. The source model is first posed to match shared
anatomical anchors, then aligned with a global similarity transform and refined
with smooth per-vertex offsets. The saved mapping contains both directions:

* ``target_to_source_*``: each target vertex as barycentric coordinates on the
  deformed source topology.
* ``source_to_target_*``: each deformed source vertex as barycentric coordinates
  on the target topology.

Usage:
    uv run scripts/mapping.py SMPL SMPLX -o smpl_to_smplx_mapping.npz --show
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, TypeAlias

from body_models.base import BodyModel
from body_models.constants import Joint
from jaxtyping import Float, Int
import loguru
import nanomanifold
import numpy as np
import rich.console
import rich.panel
import rich.table
import torch
import typer
import tqdm
import viser
import warp as wp

wp.config.quiet = True
warnings.filterwarnings(
    "ignore",
    message=r"dtype\(\): align should be passed.*",
    category=Warning,
    module=r"numpy\.lib\._format_impl",
)


app = typer.Typer(
    add_completion=False,
    help=(
        "Compute a non-rigid topological mapping between two supported body-model meshes. "
        "The target mesh is fixed; the source topology is deformed to match it."
    ),
)

MODEL_ALIASES = {
    "smpl": "SMPL",
    "smplh": "SMPLH",
    "smplx": "SMPLX",
    "mhr": "MHR",
    "soma": "SOMA",
    "garment": "Garment",
    "garments": "Garment",
    "garmentmeasurements": "Garment",
    "anny": "Anny",
    "anny3d": "Anny",
}

ANCHOR_VERTEX_NEIGHBORS = 16
SOURCE_FIT_PARAMETER_NAMES = ("shape", "identity", "scale_params", "pose", "body_pose", "hand_pose", "head_pose")
POSE_SKIP_SCALE = 0.002
FLIP_WARNING_FRACTION = 0.005
AUTO_CUDA_MIN_FREE_BYTES = 2 * 1024**3

TensorParams: TypeAlias = dict[str, torch.Tensor]
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def configure_logging(verbose: bool) -> None:
    loguru.logger.remove()
    loguru.logger.add(
        sys.stderr,
        colorize=sys.stderr.isatty(),
        format="<level>{level}: {message}</level>",
        level="INFO" if verbose else "WARNING",
    )


def progress_steps(count: int, label: str) -> tqdm.tqdm:
    return tqdm.tqdm(
        range(count),
        desc=label,
        disable=not sys.stderr.isatty(),
        dynamic_ncols=True,
        leave=False,
        mininterval=0.5,
    )


@wp.kernel
def _project_points_to_mesh(
    mesh: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    face_indices: wp.array(dtype=wp.int32),
    barycentric: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=wp.float32),
    max_dist: wp.float32,
):
    tid = wp.tid()
    query = wp.mesh_query_point_no_sign(mesh, query_points[tid], max_dist)
    if query.result:
        face_indices[tid] = query.face
        barycentric[tid] = wp.vec3(1.0 - query.u - query.v, query.u, query.v)
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        distances[tid] = wp.length(closest - query_points[tid])
    else:
        face_indices[tid] = -1
        barycentric[tid] = wp.vec3(0.0, 0.0, 0.0)
        distances[tid] = max_dist


@dataclass(frozen=True)
class MeshData:
    model_name: str
    model: BodyModel
    params: TensorParams
    vertices: Float[torch.Tensor, "vertices 3"]
    faces: np.ndarray
    joints: Float[torch.Tensor, "joints 3"]
    standard_joint_indices: dict[Joint, int]


@dataclass(frozen=True)
class Stage:
    name: str
    steps: int
    lr: float
    edge_weight: float
    laplacian_weight: float
    offset_weight: float
    normal_weight: float


@dataclass(frozen=True)
class SurfaceSamples:
    face_ids: Int[torch.Tensor, "samples"]
    weights: Float[torch.Tensor, "samples 3"]


@dataclass(frozen=True)
class SurfaceCorrespondences:
    source_to_target_faces: Int[torch.Tensor, "samples"]
    source_to_target_barycentric: Float[torch.Tensor, "samples 3"]
    target_to_source_faces: Int[torch.Tensor, "samples"]
    target_to_source_barycentric: Float[torch.Tensor, "samples 3"]


@dataclass(frozen=True)
class PoseCheckpoint:
    step: int
    loss: float
    anchor_p95: float
    pose_regularizer: float
    seconds: float


@dataclass(frozen=True)
class DeformationMetrics:
    edge_stretch_mean: float
    edge_stretch_p95: float
    edge_stretch_max: float
    offset_mean: float
    offset_p95: float
    offset_max: float


@dataclass(frozen=True)
class StageLosses:
    total: float
    fit: float
    mesh: float
    anchor: float
    edge: float
    laplacian: float
    offset: float
    normal: float


@dataclass(frozen=True)
class FitCheckpoint:
    stage: str
    step: int
    losses: StageLosses
    deformation: DeformationMetrics
    anchor_p95: float
    anchor_weight: float
    flip_fraction: float
    scale: float
    seconds: float


@dataclass(frozen=True)
class DistanceMetrics:
    mean: float
    median: float
    p95: float
    max: float
    p95_relative: float
    max_relative: float


@dataclass(frozen=True)
class MappingQuality:
    target_to_source: DistanceMetrics
    source_to_target: DistanceMetrics
    final_anchor_p95: float
    final_edge_stretch_p95: float
    final_offset_p95: float
    final_flip_fraction: float


@dataclass
class FitResult:
    vertices: Float[torch.Tensor, "vertices 3"]
    rigid_vertices: Float[torch.Tensor, "vertices 3"]
    offsets: Float[torch.Tensor, "vertices 3"]
    rotation_vector: Float[torch.Tensor, "3"]
    scale_delta: Float[torch.Tensor, ""]
    translation: Float[torch.Tensor, "3"]
    loss_history: list[FitCheckpoint]
    initial_method: str
    initial_scale: float


@dataclass(frozen=True)
class FitProblem:
    source_vertices: Float[torch.Tensor, "source_vertices 3"]
    source_faces: np.ndarray
    source_anchor_points: Float[torch.Tensor, "anchors 3"]
    target_vertices: Float[torch.Tensor, "target_vertices 3"]
    target_faces: np.ndarray
    target_anchor_points: Float[torch.Tensor, "anchors 3"]


@dataclass(frozen=True)
class FitSettings:
    stages: list[Stage]
    sample_points: int
    fit_robust_scale: float
    mesh_weight: float
    mesh_point_plane_weight: float
    mesh_query_every: int
    anchor_weight: float
    viewer: ViewerState | None
    seed: int
    update_every: int
    metric_every: int


@dataclass
class StageEvaluation:
    vertices: Float[torch.Tensor, "vertices 3"]
    rigid_vertices: Float[torch.Tensor, "vertices 3"]
    loss: Float[torch.Tensor, ""]
    fit_loss: Float[torch.Tensor, ""]
    mesh_loss: Float[torch.Tensor, ""]
    anchor_loss: Float[torch.Tensor, ""]
    anchor_p95: float
    edge_loss: Float[torch.Tensor, ""]
    smooth_loss: Float[torch.Tensor, ""]
    offset_loss: Float[torch.Tensor, ""]
    normal_loss: Float[torch.Tensor, ""]
    flip_fraction: float


@dataclass
class ViewerState:
    server: viser.ViserServer
    source: viser.MeshHandle
    status: viser.LabelHandle


def canonical_model_name(name: str) -> str:
    key = name.lower().replace("-", "").replace("_", "")
    if key not in MODEL_ALIASES:
        valid = ", ".join(["SMPL", "SMPLH", "SMPLX", "MHR", "SOMA", "Garment", "Anny"])
        raise typer.BadParameter(f"Unsupported model {name!r}. Choose one of: {valid}.")
    return MODEL_ALIASES[key]


def load_model(
    model_name: str,
    *,
    model_path: Path | None,
    gender: str | None,
    simplify: float,
    lod: int,
    soma_model_type: str,
    anny_rig: str,
    anny_topology: str,
    device: torch.device,
) -> MeshData:
    canonical = canonical_model_name(model_name)
    loguru.logger.info("Loading {} from {}", canonical, model_path or "<configured default/cache>")

    match canonical:
        case "SMPL":
            import body_models.smpl.torch as smpl_torch

            model = smpl_torch.SMPL(model_path, gender=gender, simplify=simplify)
        case "SMPLH":
            import body_models.smplh.torch as smplh_torch

            model = smplh_torch.SMPLH(model_path, gender=gender, simplify=simplify)
        case "SMPLX":
            import body_models.smplx.torch as smplx_torch

            model = smplx_torch.SMPLX(model_path, gender=gender, simplify=simplify)
        case "MHR":
            import body_models.mhr.torch as mhr_torch

            model = mhr_torch.MHR(model_path, lod=lod, simplify=simplify)
        case "SOMA":
            import body_models.soma.torch as soma_torch

            model = soma_torch.SOMA(model_path, model_type=soma_model_type, simplify=simplify)
        case "Garment":
            import body_models.garment_measurements.torch as garment_torch

            model = garment_torch.GarmentMeasurements(model_path)
        case "Anny":
            import body_models.anny.torch as anny_torch

            model = anny_torch.ANNY(model_path, rig=anny_rig, topology=anny_topology, simplify=simplify)
        case _:
            raise AssertionError(f"Unhandled model: {canonical}")

    model = model.to(device).eval()
    with torch.no_grad():
        params = model.get_rest_pose(batch_size=1, dtype=torch.float32)
        vertices, joints = forward_model(model, params)
        vertices = vertices.detach().to(device=device, dtype=torch.float32)
        joints = joints.detach().to(device=device, dtype=torch.float32)

    joint_names = list(model.joint_names)
    standard_joint_indices = {joint: model.joint_index(joint) for joint in model._standard_joints}
    if len(joint_names) != joints.shape[0]:
        raise ValueError(f"{canonical} has {len(joint_names)} joint names but {joints.shape[0]} skeleton joints.")
    faces = triangulate(to_numpy(model.faces))
    loguru.logger.info(
        "{} mesh: {} vertices, {} triangles, {} joints",
        canonical,
        vertices.shape[0],
        faces.shape[0],
        joints.shape[0],
    )
    return MeshData(
        model_name=canonical,
        model=model,
        params=params,
        vertices=vertices,
        faces=faces,
        joints=joints,
        standard_joint_indices=standard_joint_indices,
    )


def forward_model(
    model: BodyModel,
    params: TensorParams,
) -> tuple[Float[torch.Tensor, "vertices 3"], Float[torch.Tensor, "joints 3"]]:
    vertices = model.forward_vertices(**params)[0].to(dtype=torch.float32)
    skeleton = model.forward_skeleton(**params)[0].to(dtype=torch.float32)
    return vertices, skeleton[:, :3, 3]


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def common_joint_anchors(
    source: MeshData,
    target: MeshData,
) -> tuple[list[str], Float[torch.Tensor, "anchors 3"], Float[torch.Tensor, "anchors 3"]]:
    joints = common_joints(source, target)
    device = source.vertices.device
    source_index = torch.as_tensor(
        [source.standard_joint_indices[joint] for joint in joints], device=device, dtype=torch.long
    )
    target_index = torch.as_tensor(
        [target.standard_joint_indices[joint] for joint in joints], device=device, dtype=torch.long
    )
    return [joint.value for joint in joints], source.joints[source_index], target.joints[target_index]


def common_joints(source: MeshData, target: MeshData) -> list[Joint]:
    joints = [
        joint for joint in Joint if joint in source.standard_joint_indices and joint in target.standard_joint_indices
    ]
    if len(joints) < 3:
        raise ValueError(
            f"{source.model_name} and {target.model_name} share {len(joints)} standard joints; "
            "at least 3 are required for skeleton alignment."
        )
    return joints


def fit_parameters(params: TensorParams) -> tuple[TensorParams, TensorParams]:
    fit_params = dict(params)
    initial_params = {}
    for name in SOURCE_FIT_PARAMETER_NAMES:
        if name not in fit_params:
            continue
        initial = fit_params[name].detach().clone()
        fit_params[name] = torch.nn.Parameter(initial.clone())
        initial_params[name] = initial
    return fit_params, initial_params


def source_fit_regularizer(
    params: TensorParams,
    initial_params: TensorParams,
) -> Float[torch.Tensor, ""]:
    losses = [(params[name] - initial).square().mean() for name, initial in initial_params.items()]
    return torch.stack(losses).mean()


def pose_source_to_target(
    source: MeshData,
    target: MeshData,
    *,
    steps: int,
    lr: float,
    regularizer: float,
    metric_every: int,
) -> tuple[MeshData, list[PoseCheckpoint]]:
    if steps <= 0:
        return source, []

    anchor_joints = common_joints(source, target)
    pose_params, initial_params = fit_parameters(source.params)
    if not initial_params:
        raise ValueError(f"{source.model_name} exposes no optimizable parameters for source fitting.")

    device = source.vertices.device
    source_indices = [source.standard_joint_indices[joint] for joint in anchor_joints]
    target_indices = [target.standard_joint_indices[joint] for joint in anchor_joints]
    source_index = torch.as_tensor(source_indices, device=device, dtype=torch.long)
    target_index = torch.as_tensor(target_indices, device=device, dtype=torch.long)
    target_anchors = target.joints[target_index]
    source_anchors = source.joints[source_index]
    aligned_anchors, _ = paired_similarity(source_anchors, target_anchors, source_anchors)
    initial_distances = torch.linalg.vector_norm(aligned_anchors - target_anchors, dim=-1)
    initial_p95 = float(torch.quantile(initial_distances, 0.95).cpu())
    target_size = torch.linalg.vector_norm(target.vertices.max(dim=0).values - target.vertices.min(dim=0).values).item()
    if initial_p95 <= target_size * POSE_SKIP_SCALE:
        loguru.logger.info("Skipping source pose optimization; initial anchor p95 is {:.2f} mm", initial_p95 * 1000.0)
        return source, []

    optimizer = torch.optim.Adam([pose_params[name] for name in initial_params], lr=lr)
    history: list[PoseCheckpoint] = []

    loguru.logger.info("Optimizing source pose for {} steps using {} anchors", steps, len(anchor_joints))
    started_at = time.perf_counter()
    progress = progress_steps(steps, "source fit")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        _, joints = forward_model(source.model, pose_params)
        source_anchors = joints[source_index]
        aligned_anchors, _ = paired_similarity(source_anchors, target_anchors, source_anchors)
        distances = torch.linalg.vector_norm(aligned_anchors - target_anchors, dim=-1)
        anchor_loss = distances.square().mean()
        reg_loss = source_fit_regularizer(pose_params, initial_params)
        loss = anchor_loss + regularizer * reg_loss
        loss.backward()
        optimizer.step()

        should_report = step == 0 or step == steps - 1 or (step + 1) % metric_every == 0
        if should_report:
            anchor_p95 = float(torch.quantile(distances.detach(), 0.95).cpu())
            row = PoseCheckpoint(
                step=step + 1,
                loss=float(loss.detach().cpu()),
                anchor_p95=anchor_p95,
                pose_regularizer=float(reg_loss.detach().cpu()),
                seconds=time.perf_counter() - started_at,
            )
            history.append(row)
            progress.set_postfix(anc95=f"{anchor_p95 * 1000.0:.1f}mm")

    final_params = {name: value.detach().clone() for name, value in pose_params.items()}
    with torch.no_grad():
        vertices, joints = forward_model(source.model, final_params)
    loguru.logger.info("Source pose anchor p95: {:.2f} mm", history[-1].anchor_p95 * 1000.0)
    return (
        MeshData(
            model_name=source.model_name,
            model=source.model,
            params=final_params,
            vertices=vertices.detach(),
            faces=source.faces,
            joints=joints.detach(),
            standard_joint_indices=source.standard_joint_indices,
        ),
        history,
    )


def triangulate(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2:
        raise ValueError(f"faces must have shape [F, 3] or [F, 4], got {faces.shape}")
    if faces.shape[1] == 3:
        return np.ascontiguousarray(faces)
    if faces.shape[1] == 4:
        tris = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)
        return np.ascontiguousarray(tris)
    raise ValueError(f"Unsupported face arity: {faces.shape[1]}")


def unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0).astype(np.int64)


def mesh_area_probabilities(
    vertices: Float[torch.Tensor, "vertices 3"],
    faces: Int[torch.Tensor, "faces 3"],
) -> Float[torch.Tensor, "faces"]:
    tri = vertices[faces]
    edge_01 = tri[:, 1] - tri[:, 0]
    edge_02 = tri[:, 2] - tri[:, 0]
    normals = torch.cross(edge_01, edge_02, dim=-1)
    areas = 0.5 * torch.linalg.vector_norm(normals, dim=-1)
    areas = torch.clamp(areas, min=torch.finfo(vertices.dtype).eps)
    return areas / areas.sum()


def make_surface_samples(
    face_probabilities: Float[torch.Tensor, "faces"],
    sample_count: int,
) -> SurfaceSamples:
    face_ids = torch.multinomial(face_probabilities, sample_count, replacement=True)
    random = torch.rand((sample_count, 2), device=face_probabilities.device, dtype=face_probabilities.dtype)
    sqrt_u = torch.sqrt(random[:, 0])
    w0 = 1.0 - sqrt_u
    w1 = sqrt_u * (1.0 - random[:, 1])
    w2 = sqrt_u * random[:, 1]
    weights = torch.stack([w0, w1, w2], dim=1)
    return SurfaceSamples(face_ids=face_ids, weights=weights)


def sample_surface(
    vertices: Float[torch.Tensor, "vertices 3"],
    faces: Int[torch.Tensor, "faces 3"],
    samples: SurfaceSamples,
) -> Float[torch.Tensor, "samples 3"]:
    tri = vertices[faces[samples.face_ids]]
    weights = samples.weights[:, :, None]
    return torch.sum(tri * weights, dim=1)


def closest_points_on_faces(
    vertices: Float[torch.Tensor, "vertices 3"],
    faces: Int[torch.Tensor, "faces 3"],
    face_ids: Int[torch.Tensor, "samples"],
    barycentric: Float[torch.Tensor, "samples 3"],
) -> Float[torch.Tensor, "samples 3"]:
    triangles = vertices[faces[face_ids]]
    return torch.sum(triangles * barycentric[:, :, None], dim=1)


def chamfer_loss(
    source_points: Float[torch.Tensor, "source_samples 3"],
    target_points: Float[torch.Tensor, "target_samples 3"],
    robust_scale: float,
) -> Float[torch.Tensor, ""]:
    distances = torch.cdist(source_points, target_points)
    source_to_target = robust_mean_square(distances.min(dim=1).values, robust_scale)
    target_to_source = robust_mean_square(distances.min(dim=0).values, robust_scale)
    return 0.5 * (source_to_target + target_to_source)


def mesh_surface_loss(
    source_samples: Float[torch.Tensor, "samples 3"],
    target_samples: Float[torch.Tensor, "samples 3"],
    vertices: Float[torch.Tensor, "vertices 3"],
    source_faces: Int[torch.Tensor, "faces 3"],
    target_vertices: Float[torch.Tensor, "target_vertices 3"],
    target_faces: Int[torch.Tensor, "target_faces 3"],
    correspondences: SurfaceCorrespondences,
    robust_scale: float,
    point_plane_weight: float,
) -> Float[torch.Tensor, ""]:
    target_closest = closest_points_on_faces(
        target_vertices,
        target_faces,
        correspondences.source_to_target_faces,
        correspondences.source_to_target_barycentric,
    )
    source_closest = closest_points_on_faces(
        vertices,
        source_faces,
        correspondences.target_to_source_faces,
        correspondences.target_to_source_barycentric,
    )
    source_residuals = source_samples - target_closest
    target_residuals = target_samples - source_closest
    point_loss = 0.5 * (
        robust_mean_square(torch.linalg.vector_norm(source_residuals, dim=-1), robust_scale)
        + robust_mean_square(torch.linalg.vector_norm(target_residuals, dim=-1), robust_scale)
    )

    target_normals = face_normals(target_vertices, target_faces)[correspondences.source_to_target_faces].detach()
    source_normals = face_normals(vertices, source_faces)[correspondences.target_to_source_faces].detach()
    plane_loss = 0.5 * (
        robust_mean_square(torch.sum(source_residuals * target_normals, dim=-1), robust_scale)
        + robust_mean_square(torch.sum(target_residuals * source_normals, dim=-1), robust_scale)
    )
    return point_loss + point_plane_weight * plane_loss


def robust_mean_square(values: Float[torch.Tensor, "values"], scale: float) -> Float[torch.Tensor, ""]:
    if scale <= 0.0:
        return values.square().mean()
    scale_tensor = values.new_tensor(scale)
    normalized = values / scale_tensor
    return 2.0 * scale_tensor.square() * (torch.sqrt(1.0 + normalized.square()) - 1.0).mean()


def tensor_metrics(values: Float[torch.Tensor, "values"]) -> tuple[float, float, float]:
    values = values.detach()
    mean = float(values.mean().cpu())
    p95 = float(torch.quantile(values, 0.95).cpu())
    maximum = float(values.max().cpu())
    return mean, p95, maximum


def initial_similarity(
    source_vertices: Float[torch.Tensor, "source_vertices 3"],
    target_vertices: Float[torch.Tensor, "target_vertices 3"],
) -> tuple[Float[torch.Tensor, "source_vertices 3"], Float[torch.Tensor, "3"], Float[torch.Tensor, "3"], float]:
    source_center = source_vertices.mean(dim=0)
    target_center = target_vertices.mean(dim=0)
    source_radius = torch.sqrt((source_vertices - source_center).square().sum(dim=-1).mean())
    target_radius = torch.sqrt((target_vertices - target_center).square().sum(dim=-1).mean())
    scale = (target_radius / torch.clamp(source_radius, min=1.0e-8)).item()
    source_base = (source_vertices - source_center) * scale
    return source_base + target_center, source_center, target_center, scale


def paired_similarity(
    source_points: Float[torch.Tensor, "anchors 3"],
    target_points: Float[torch.Tensor, "anchors 3"],
    points: Float[torch.Tensor, "points 3"],
) -> tuple[Float[torch.Tensor, "points 3"], float]:
    source_center = source_points.mean(dim=0)
    target_center = target_points.mean(dim=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    covariance = source_centered.T @ target_centered / source_points.shape[0]
    u, singular_values, vh = torch.linalg.svd(covariance)
    sign = torch.ones(3, device=points.device, dtype=points.dtype)
    if torch.linalg.det(vh.T @ u.T) < 0:
        sign[-1] = -1.0
    rotation = vh.T @ torch.diag(sign) @ u.T
    variance = source_centered.square().sum(dim=-1).mean()
    scale = (singular_values * sign).sum() / torch.clamp(variance, min=1.0e-8)
    aligned = scale * ((points - source_center) @ rotation.T) + target_center
    return aligned, float(scale.detach().cpu())


def initial_alignment(
    source_vertices: Float[torch.Tensor, "source_vertices 3"],
    target_vertices: Float[torch.Tensor, "target_vertices 3"],
    source_anchors: Float[torch.Tensor, "anchors 3"],
    target_anchors: Float[torch.Tensor, "anchors 3"],
) -> tuple[
    Float[torch.Tensor, "source_vertices 3"],
    Float[torch.Tensor, "anchors 3"],
    Float[torch.Tensor, "3"],
    float,
    str,
]:
    if source_anchors.shape[0] < 3:
        raise ValueError("At least 3 skeleton anchors are required for initial alignment.")
    target_center = target_vertices.mean(dim=0)
    aligned_vertices, scale = paired_similarity(source_anchors, target_anchors, source_vertices)
    aligned_anchors, _ = paired_similarity(source_anchors, target_anchors, source_anchors)
    return aligned_vertices, aligned_anchors, target_center, scale, "skeleton"


def rigid_transform(
    points: Float[torch.Tensor, "points 3"],
    target_center: Float[torch.Tensor, "3"],
    rotation_vector: Float[torch.Tensor, "3"],
    scale_delta: Float[torch.Tensor, ""],
    translation: Float[torch.Tensor, "3"],
) -> Float[torch.Tensor, "points 3"]:
    centered = points - target_center
    return rotate_vectors(centered, rotation_vector, scale_delta) + target_center + translation


def rotate_vectors(
    vectors: Float[torch.Tensor, "points 3"],
    rotation_vector: Float[torch.Tensor, "3"],
    scale_delta: Float[torch.Tensor, ""],
) -> Float[torch.Tensor, "points 3"]:
    quaternion = nanomanifold.SO3.exp(rotation_vector, xp=torch)
    rotation = nanomanifold.SO3.to_rotmat(quaternion, xp=torch)
    return torch.exp(scale_delta) * (vectors @ rotation.T)


def transformed_source(
    source_aligned: Float[torch.Tensor, "vertices 3"],
    target_center: Float[torch.Tensor, "3"],
    rotation_vector: Float[torch.Tensor, "3"],
    scale_delta: Float[torch.Tensor, ""],
    translation: Float[torch.Tensor, "3"],
    offsets: Float[torch.Tensor, "vertices 3"],
) -> tuple[Float[torch.Tensor, "vertices 3"], Float[torch.Tensor, "vertices 3"]]:
    rigid = rigid_transform(source_aligned, target_center, rotation_vector, scale_delta, translation)
    return rigid + offsets, rigid


def edge_stretch_loss(
    vertices: Float[torch.Tensor, "vertices 3"],
    edges: Int[torch.Tensor, "edges 2"],
    rest_edge_lengths: Float[torch.Tensor, "edges"],
) -> Float[torch.Tensor, ""]:
    lengths = torch.linalg.vector_norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], dim=-1)
    stretch = (lengths - rest_edge_lengths) / torch.clamp(rest_edge_lengths, min=1.0e-8)
    return robust_mean_square(stretch, 0.1)


def smooth_offset_loss(
    offsets: Float[torch.Tensor, "vertices 3"],
    edges: Int[torch.Tensor, "edges 2"],
) -> Float[torch.Tensor, ""]:
    return (offsets[edges[:, 0]] - offsets[edges[:, 1]]).square().sum(dim=-1).mean()


def face_normals(
    vertices: Float[torch.Tensor, "vertices 3"],
    faces: Int[torch.Tensor, "faces 3"],
) -> Float[torch.Tensor, "faces 3"]:
    tri = vertices[faces]
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    return torch.nn.functional.normalize(normals, dim=-1, eps=1.0e-8)


def normal_consistency_loss(
    vertices: Float[torch.Tensor, "vertices 3"],
    rigid_vertices: Float[torch.Tensor, "vertices 3"],
    faces: Int[torch.Tensor, "faces 3"],
) -> Float[torch.Tensor, ""]:
    normals = face_normals(vertices, faces)
    rigid_normals = face_normals(rigid_vertices.detach(), faces)
    cosine = torch.sum(normals * rigid_normals, dim=-1).clamp(-1.0, 1.0)
    return robust_mean_square(1.0 - cosine, 0.1)


def face_flip_fraction(
    vertices: Float[torch.Tensor, "vertices 3"],
    rigid_vertices: Float[torch.Tensor, "vertices 3"],
    faces: Int[torch.Tensor, "faces 3"],
) -> float:
    with torch.no_grad():
        normals = face_normals(vertices, faces)
        rigid_normals = face_normals(rigid_vertices, faces)
        cosine = torch.sum(normals * rigid_normals, dim=-1)
        return float((cosine < 0.0).to(dtype=torch.float32).mean().cpu())


def anchor_vertex_attachment(
    anchor_points: Float[torch.Tensor, "anchors 3"],
    vertices: Float[torch.Tensor, "vertices 3"],
) -> tuple[
    Int[torch.Tensor, "anchors neighbors"],
    Float[torch.Tensor, "anchors neighbors"],
    Float[torch.Tensor, "anchors 3"],
]:
    neighbor_count = min(ANCHOR_VERTEX_NEIGHBORS, vertices.shape[0])
    distances, indices = torch.topk(torch.cdist(anchor_points, vertices), neighbor_count, largest=False)
    weights = 1.0 / torch.clamp(distances, min=1.0e-4).square()
    weights = weights / weights.sum(dim=1, keepdim=True)
    local_points = torch.sum(vertices[indices] * weights[:, :, None], dim=1)
    return indices, weights, anchor_points - local_points


def attached_anchor_points(
    vertices: Float[torch.Tensor, "vertices 3"],
    indices: Int[torch.Tensor, "anchors neighbors"],
    weights: Float[torch.Tensor, "anchors neighbors"],
    local_offsets: Float[torch.Tensor, "anchors 3"],
    rotation_vector: Float[torch.Tensor, "3"],
    scale_delta: Float[torch.Tensor, ""],
) -> Float[torch.Tensor, "anchors 3"]:
    surface_points = torch.sum(vertices[indices] * weights[:, :, None], dim=1)
    return surface_points + rotate_vectors(local_offsets, rotation_vector, scale_delta)


def deformation_metrics(
    vertices: Float[torch.Tensor, "vertices 3"],
    offsets: Float[torch.Tensor, "vertices 3"],
    edges: Int[torch.Tensor, "edges 2"],
    rest_edge_lengths: Float[torch.Tensor, "edges"],
    scale_delta: Float[torch.Tensor, ""],
) -> DeformationMetrics:
    edge_lengths = torch.linalg.vector_norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], dim=-1)
    expected_lengths = rest_edge_lengths * torch.exp(scale_delta.detach())
    relative_stretch = torch.abs(edge_lengths - expected_lengths) / torch.clamp(expected_lengths, min=1.0e-8)
    offset_norms = torch.linalg.vector_norm(offsets, dim=-1)
    edge_stretch_mean, edge_stretch_p95, edge_stretch_max = tensor_metrics(relative_stretch)
    offset_mean, offset_p95, offset_max = tensor_metrics(offset_norms)
    return DeformationMetrics(
        edge_stretch_mean=edge_stretch_mean,
        edge_stretch_p95=edge_stretch_p95,
        edge_stretch_max=edge_stretch_max,
        offset_mean=offset_mean,
        offset_p95=offset_p95,
        offset_max=offset_max,
    )


def build_viewer(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    port: int,
) -> ViewerState:
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/grid", plane="xz")
    server.scene.add_mesh_simple("/target", vertices=target_vertices, faces=target_faces, color=(180, 180, 180))
    source = server.scene.add_mesh_simple(
        "/source_deformed",
        vertices=source_vertices,
        faces=source_faces,
        color=(70, 140, 240),
    )
    label_y = float(target_vertices[:, 1].max()) + 0.15
    status = server.scene.add_label("/status", text="initializing", position=(0.0, label_y, 0.0))
    loguru.logger.info("Viser server running at http://localhost:{}", port)
    return ViewerState(server=server, source=source, status=status)


def update_viewer(viewer: ViewerState | None, vertices: Float[torch.Tensor, "vertices 3"], status: str) -> None:
    if viewer is None:
        return
    viewer.source.vertices = vertices.detach().cpu().numpy().astype(np.float32)
    viewer.status.text = status


def mesh_correspondences(
    *,
    source_samples: Float[torch.Tensor, "samples 3"],
    target_samples: Float[torch.Tensor, "samples 3"],
    vertices: Float[torch.Tensor, "vertices 3"],
    source_faces: np.ndarray,
    target_vertices: Float[torch.Tensor, "target_vertices 3"],
    target_faces: np.ndarray,
    torch_device: torch.device,
) -> SurfaceCorrespondences:
    source_to_target_faces, source_to_target_barycentric, _ = project_points_with_warp(
        mesh_vertices=to_numpy(target_vertices),
        mesh_faces=target_faces,
        query_points=to_numpy(source_samples),
        torch_device=torch_device,
    )
    target_to_source_faces, target_to_source_barycentric, _ = project_points_with_warp(
        mesh_vertices=to_numpy(vertices),
        mesh_faces=source_faces,
        query_points=to_numpy(target_samples),
        torch_device=torch_device,
    )
    device = vertices.device
    return SurfaceCorrespondences(
        source_to_target_faces=torch.as_tensor(source_to_target_faces, device=device, dtype=torch.long),
        source_to_target_barycentric=torch.as_tensor(source_to_target_barycentric, device=device, dtype=torch.float32),
        target_to_source_faces=torch.as_tensor(target_to_source_faces, device=device, dtype=torch.long),
        target_to_source_barycentric=torch.as_tensor(target_to_source_barycentric, device=device, dtype=torch.float32),
    )


def fit_source_to_target(problem: FitProblem, settings: FitSettings) -> FitResult:
    torch.manual_seed(settings.seed)
    source_vertices = problem.source_vertices
    target_vertices = problem.target_vertices
    device = source_vertices.device
    source_faces = torch.as_tensor(problem.source_faces, device=device, dtype=torch.long)
    target_faces = torch.as_tensor(problem.target_faces, device=device, dtype=torch.long)
    source_edges = torch.as_tensor(unique_edges_from_faces(problem.source_faces), device=device, dtype=torch.long)
    target_size = torch.linalg.vector_norm(target_vertices.max(dim=0).values - target_vertices.min(dim=0).values).item()
    fit_robust_scale = target_size * settings.fit_robust_scale

    source_aligned, source_anchors, target_center, initial_scale, initial_method = initial_alignment(
        source_vertices,
        target_vertices,
        problem.source_anchor_points,
        problem.target_anchor_points,
    )
    anchor_indices, anchor_weights, anchor_local_offsets = anchor_vertex_attachment(
        source_anchors,
        source_aligned,
    )
    source_area_prob = mesh_area_probabilities(source_aligned, source_faces)
    target_area_prob = mesh_area_probabilities(target_vertices, target_faces)

    rotation_vector = torch.nn.Parameter(torch.zeros(3, device=device, dtype=torch.float32))
    scale_delta = torch.nn.Parameter(torch.zeros((), device=device, dtype=torch.float32))
    translation = torch.nn.Parameter(torch.zeros(3, device=device, dtype=torch.float32))
    offsets = torch.nn.Parameter(torch.zeros_like(source_aligned))
    rest_edge_lengths = torch.linalg.vector_norm(
        source_aligned[source_edges[:, 0]] - source_aligned[source_edges[:, 1]], dim=-1
    )

    loguru.logger.info("Initial {} scale applied to source: {:.6g}", initial_method, initial_scale)
    loss_history: list[FitCheckpoint] = []
    vertices, _ = transformed_source(
        source_aligned,
        target_center,
        rotation_vector,
        scale_delta,
        translation,
        offsets,
    )
    update_viewer(settings.viewer, vertices, "initial alignment")

    for stage in settings.stages:
        if stage.steps <= 0:
            continue
        anchor_loss_weight = settings.anchor_weight if stage.name == "rigid" else 0.0
        optimize_offsets = stage.name != "rigid"
        offsets.requires_grad_(optimize_offsets)
        params = [rotation_vector, scale_delta, translation]
        if optimize_offsets:
            params.append(offsets)
        optimizer = torch.optim.Adam(params, lr=stage.lr)
        source_surface_samples = make_surface_samples(source_area_prob, settings.sample_points)
        target_surface_samples = make_surface_samples(target_area_prob, settings.sample_points)
        target_samples = sample_surface(target_vertices, target_faces, target_surface_samples)
        correspondences: SurfaceCorrespondences | None = None

        def evaluate_stage(refresh_mesh: bool) -> StageEvaluation:
            nonlocal correspondences
            vertices, rigid_vertices = transformed_source(
                source_aligned,
                target_center,
                rotation_vector,
                scale_delta,
                translation,
                offsets,
            )
            source_samples = sample_surface(vertices, source_faces, source_surface_samples)
            chamfer = chamfer_loss(source_samples, target_samples, fit_robust_scale)
            if settings.mesh_weight > 0.0:
                if refresh_mesh or correspondences is None:
                    correspondences = mesh_correspondences(
                        source_samples=source_samples,
                        target_samples=target_samples,
                        vertices=vertices,
                        source_faces=problem.source_faces,
                        target_vertices=target_vertices,
                        target_faces=problem.target_faces,
                        torch_device=device,
                    )
                mesh_loss = mesh_surface_loss(
                    source_samples,
                    target_samples,
                    vertices,
                    source_faces,
                    target_vertices,
                    target_faces,
                    correspondences,
                    fit_robust_scale,
                    settings.mesh_point_plane_weight,
                )
            else:
                mesh_loss = vertices.new_zeros(())
            fit_loss = chamfer + settings.mesh_weight * mesh_loss
            moved_anchors = attached_anchor_points(
                vertices,
                anchor_indices,
                anchor_weights,
                anchor_local_offsets,
                rotation_vector,
                scale_delta,
            )
            anchor_distances = torch.linalg.vector_norm(moved_anchors - problem.target_anchor_points, dim=-1)
            anchor_loss = anchor_distances.square().mean()
            anchor_p95 = float(torch.quantile(anchor_distances.detach(), 0.95).cpu())
            current_edge_rest = rest_edge_lengths * torch.exp(scale_delta.detach())
            edge_loss = edge_stretch_loss(vertices, source_edges, current_edge_rest)
            smooth_loss = smooth_offset_loss(offsets, source_edges)
            offset_loss = offsets.square().sum(dim=-1).mean()
            normal_loss = normal_consistency_loss(vertices, rigid_vertices, source_faces)
            flip_fraction = face_flip_fraction(vertices, rigid_vertices, source_faces)
            loss = (
                fit_loss
                + anchor_loss_weight * anchor_loss
                + stage.edge_weight * edge_loss
                + stage.laplacian_weight * smooth_loss
                + stage.offset_weight * offset_loss
                + stage.normal_weight * normal_loss
            )
            return StageEvaluation(
                vertices=vertices,
                rigid_vertices=rigid_vertices,
                loss=loss,
                fit_loss=fit_loss,
                mesh_loss=mesh_loss,
                anchor_loss=anchor_loss,
                anchor_p95=anchor_p95,
                edge_loss=edge_loss,
                smooth_loss=smooth_loss,
                offset_loss=offset_loss,
                normal_loss=normal_loss,
                flip_fraction=flip_fraction,
            )

        loguru.logger.info("Starting {} stage for {} steps", stage.name, stage.steps)
        stage_started_at = time.perf_counter()
        progress = progress_steps(stage.steps, stage.name)
        for step in progress:
            optimizer.zero_grad(set_to_none=True)
            refresh_mesh = step % settings.mesh_query_every == 0
            evaluation = evaluate_stage(refresh_mesh=refresh_mesh)
            evaluation.loss.backward()
            optimizer.step()

            should_report = step == 0 or step == stage.steps - 1 or (step + 1) % settings.metric_every == 0
            should_update = step % settings.update_every == 0 or step == stage.steps - 1
            if should_report or should_update:
                with torch.no_grad():
                    evaluation = evaluate_stage(refresh_mesh=False)
            if should_report:
                deformation = deformation_metrics(
                    evaluation.vertices, offsets, source_edges, rest_edge_lengths, scale_delta
                )
                losses = StageLosses(
                    total=float(evaluation.loss.detach().cpu()),
                    fit=float(evaluation.fit_loss.detach().cpu()),
                    mesh=float(evaluation.mesh_loss.detach().cpu()),
                    anchor=float(evaluation.anchor_loss.detach().cpu()),
                    edge=float(evaluation.edge_loss.detach().cpu()),
                    laplacian=float(evaluation.smooth_loss.detach().cpu()),
                    offset=float(evaluation.offset_loss.detach().cpu()),
                    normal=float(evaluation.normal_loss.detach().cpu()),
                )
                row = FitCheckpoint(
                    stage=stage.name,
                    step=step + 1,
                    losses=losses,
                    deformation=deformation,
                    anchor_p95=evaluation.anchor_p95,
                    anchor_weight=anchor_loss_weight,
                    flip_fraction=evaluation.flip_fraction,
                    scale=float(torch.exp(scale_delta.detach()).cpu()),
                    seconds=time.perf_counter() - stage_started_at,
                )
                loss_history.append(row)
                progress.set_postfix(
                    fit=f"{row.losses.fit:.2e}",
                    anc95=f"{row.anchor_p95 * 1000.0:.1f}mm",
                    edge95=f"{row.deformation.edge_stretch_p95:.2%}",
                    off95=f"{row.deformation.offset_p95 * 1000.0:.1f}mm",
                    flip=f"{row.flip_fraction:.2%}",
                )
            if should_update:
                update_viewer(settings.viewer, evaluation.vertices, f"{stage.name} {step + 1}/{stage.steps}")

        loguru.logger.info(
            "{} metrics: fit={:.3e}, anchor_p95={:.2f} mm, edge_p95={:.2%}, offset_p95={:.2f} mm, {:.1f} steps/s",
            stage.name,
            loss_history[-1].losses.fit,
            loss_history[-1].anchor_p95 * 1000.0,
            loss_history[-1].deformation.edge_stretch_p95,
            loss_history[-1].deformation.offset_p95 * 1000.0,
            stage.steps / max(loss_history[-1].seconds, 1.0e-6),
        )

    final_vertices, final_rigid = transformed_source(
        source_aligned,
        target_center,
        rotation_vector,
        scale_delta,
        translation,
        offsets,
    )
    update_viewer(settings.viewer, final_vertices, "final projection")
    return FitResult(
        vertices=final_vertices.detach(),
        rigid_vertices=final_rigid.detach(),
        offsets=offsets.detach(),
        rotation_vector=rotation_vector.detach(),
        scale_delta=scale_delta.detach(),
        translation=translation.detach(),
        loss_history=loss_history,
        initial_method=initial_method,
        initial_scale=initial_scale,
    )


def warp_device_for(torch_device: torch.device) -> str:
    if torch_device.type == "cuda":
        index = torch_device.index if torch_device.index is not None else 0
        return f"cuda:{index}"
    return "cpu"


def project_points_with_warp(
    *,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    query_points: np.ndarray,
    torch_device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wp.init()
    wp_device = warp_device_for(torch_device)
    mesh_vertices = np.ascontiguousarray(mesh_vertices, dtype=np.float32)
    mesh_faces = np.ascontiguousarray(mesh_faces, dtype=np.int32)
    query_points = np.ascontiguousarray(query_points, dtype=np.float32)

    bbox = mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0)
    max_dist = np.float32(np.linalg.norm(bbox) * 10.0)
    points_wp = wp.array(mesh_vertices, dtype=wp.vec3, device=wp_device)
    indices_wp = wp.array(mesh_faces.reshape(-1), dtype=wp.int32, device=wp_device)
    queries_wp = wp.array(query_points, dtype=wp.vec3, device=wp_device)
    face_indices_wp = wp.empty(query_points.shape[0], dtype=wp.int32, device=wp_device)
    barycentric_wp = wp.empty(query_points.shape[0], dtype=wp.vec3, device=wp_device)
    distances_wp = wp.empty(query_points.shape[0], dtype=wp.float32, device=wp_device)
    mesh = wp.Mesh(points_wp, indices_wp)

    wp.launch(
        _project_points_to_mesh,
        dim=query_points.shape[0],
        inputs=[mesh.id, queries_wp, face_indices_wp, barycentric_wp, distances_wp, max_dist],
        device=wp_device,
    )
    wp.synchronize_device(wp_device)
    face_indices = face_indices_wp.numpy().astype(np.int32, copy=False)
    barycentric = barycentric_wp.numpy().astype(np.float32, copy=False)
    distances = distances_wp.numpy().astype(np.float32, copy=False)
    if np.any(face_indices < 0):
        misses = int(np.count_nonzero(face_indices < 0))
        raise RuntimeError(f"Warp failed to project {misses} query points within max_dist={float(max_dist):.4g}.")
    return face_indices, barycentric, distances


def distance_metrics(distances: np.ndarray, scale: float) -> DistanceMetrics:
    mean = float(np.mean(distances))
    median = float(np.percentile(distances, 50))
    p95 = float(np.percentile(distances, 95))
    maximum = float(np.max(distances))
    return DistanceMetrics(
        mean=mean,
        median=median,
        p95=p95,
        max=maximum,
        p95_relative=p95 / scale,
        max_relative=maximum / scale,
    )


def quality_warnings(quality: MappingQuality, target_scale: float, anchor_count: int) -> list[str]:
    edge_p95 = quality.final_edge_stretch_p95
    offset_p95 = quality.final_offset_p95 / target_scale

    warnings = []
    if anchor_count < 8:
        warnings.append(f"Only {anchor_count} skeleton anchors matched; semantic alignment may be underconstrained.")
    if max(quality.target_to_source.p95_relative, quality.source_to_target.p95_relative) > 0.005:
        warnings.append(
            "Projection p95 is high relative to the target size; inspect coverage or use a closer source topology."
        )
    if max(quality.target_to_source.max_relative, quality.source_to_target.max_relative) > 0.02:
        warnings.append("Projection max distance is high; outlier regions may be unmatched.")
    if edge_p95 > 0.05:
        warnings.append("Edge stretch p95 is high; the source mesh may be distorting to chase the target.")
    if quality.final_flip_fraction > FLIP_WARNING_FRACTION:
        warnings.append("Some source triangles flipped relative to the rigid source; inspect the deformation.")
    if offset_p95 > 0.01:
        warnings.append("Offset p95 is high relative to the target size; the result may be over-deformed.")
    if quality.final_anchor_p95 / target_scale > 0.01:
        warnings.append("Skeleton anchor p95 is high; semantic alignment may be poor.")
    return warnings


def print_summary(
    quality: MappingQuality,
    warnings: list[str],
    history: list[FitCheckpoint],
    pose_history: list[PoseCheckpoint],
    output: Path,
    elapsed_seconds: float,
) -> None:
    table = rich.table.Table(title="Mapping quality")
    table.add_column("Proxy")
    table.add_column("Value", justify="right")
    table.add_row("target -> source p95", f"{quality.target_to_source.p95 * 1000.0:.2f} mm")
    table.add_row("source -> target p95", f"{quality.source_to_target.p95 * 1000.0:.2f} mm")
    table.add_row("target -> source max", f"{quality.target_to_source.max * 1000.0:.2f} mm")
    table.add_row("source -> target max", f"{quality.source_to_target.max * 1000.0:.2f} mm")
    table.add_row("skeleton anchor p95", f"{quality.final_anchor_p95 * 1000.0:.2f} mm")
    table.add_row("edge stretch p95", f"{quality.final_edge_stretch_p95:.2%}")
    table.add_row("offset p95", f"{quality.final_offset_p95 * 1000.0:.2f} mm")
    table.add_row("triangle flips", f"{quality.final_flip_fraction:.2%}")

    stages = rich.table.Table(title="Optimization checkpoints")
    stages.add_column("Stage")
    stages.add_column("Anchor p95", justify="right")
    stages.add_column("Edge p95", justify="right")
    stages.add_column("Offset p95", justify="right")
    stages.add_column("Flips", justify="right")
    for stage in dict.fromkeys(row.stage for row in history):
        rows = [row for row in history if row.stage == stage]
        if not rows:
            continue
        row = rows[-1]
        stages.add_row(
            stage,
            f"{row.anchor_p95 * 1000.0:.2f} mm",
            f"{row.deformation.edge_stretch_p95:.2%}",
            f"{row.deformation.offset_p95 * 1000.0:.2f} mm",
            f"{row.flip_fraction:.2%}",
        )

    console = rich.console.Console()
    if pose_history:
        pose = pose_history[-1]
        pose_table = rich.table.Table(title="Source parameter fit")
        pose_table.add_column("Anchor p95", justify="right")
        pose_table.add_column("Fit reg", justify="right")
        pose_table.add_column("Speed", justify="right")
        pose_table.add_row(
            f"{pose.anchor_p95 * 1000.0:.2f} mm",
            f"{pose.pose_regularizer:.3e}",
            f"{pose.step / pose.seconds:.1f} steps/s",
        )
        console.print(pose_table)
    console.print(table)
    console.print(stages)
    if warnings:
        warning_text = "\n".join(f"- {warning}" for warning in warnings)
        console.print(rich.panel.Panel(warning_text, title="Quality warnings", style="yellow"))
    console.print(f"Saved [bold]{output}[/bold] in {elapsed_seconds:.1f}s")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if not torch.cuda.is_available():
            raise typer.BadParameter("CUDA is unavailable for --device auto; pass --device cpu to run on CPU.")
        free_bytes = cuda_free_bytes()
        if free_bytes < AUTO_CUDA_MIN_FREE_BYTES:
            free_gb = free_bytes / 1024**3
            required_gb = AUTO_CUDA_MIN_FREE_BYTES / 1024**3
            raise typer.BadParameter(
                f"CUDA has only {free_gb:.2f} GiB free for --device auto; "
                f"{required_gb:.2f} GiB is required. Pass --device cpu to run on CPU."
            )
        return torch.device("cuda:0")
    if device == "cpu":
        hide_cuda_from_warp()
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise typer.BadParameter("CUDA was requested, but torch.cuda.is_available() is false.")
    return resolved


def cuda_free_bytes() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    first_line = result.stdout.splitlines()[0]
    return int(first_line.strip()) * 1024**2


def hide_cuda_from_warp() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def save_mapping(
    *,
    output: Path,
    source: MeshData,
    target: MeshData,
    fit: FitResult,
    target_to_source_faces: np.ndarray,
    target_to_source_barycentric: np.ndarray,
    target_to_source_distances: np.ndarray,
    source_to_target_faces: np.ndarray,
    source_to_target_barycentric: np.ndarray,
    source_to_target_distances: np.ndarray,
    metadata: dict[str, JsonValue],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    source_faces = np.ascontiguousarray(source.faces, dtype=np.int32)
    target_faces = np.ascontiguousarray(target.faces, dtype=np.int32)
    source_face_vertex_indices = source_faces[target_to_source_faces]
    target_face_vertex_indices = target_faces[source_to_target_faces]
    source_fit = {
        f"source_fit_{name}": to_numpy(value).astype(np.float32)
        for name, value in source.params.items()
        if name in SOURCE_FIT_PARAMETER_NAMES
    }
    np.savez_compressed(
        output,
        source_model=np.array(source.model_name),
        target_model=np.array(target.model_name),
        source_vertices=to_numpy(source.vertices).astype(np.float32),
        source_faces=source_faces,
        target_vertices=to_numpy(target.vertices).astype(np.float32),
        target_faces=target_faces,
        deformed_source_vertices=to_numpy(fit.vertices).astype(np.float32),
        rigid_source_vertices=to_numpy(fit.rigid_vertices).astype(np.float32),
        source_offsets=to_numpy(fit.offsets).astype(np.float32),
        target_to_source_face_indices=target_to_source_faces.astype(np.int32),
        target_to_source_vertex_indices=source_face_vertex_indices.astype(np.int32),
        target_to_source_barycentric=target_to_source_barycentric.astype(np.float32),
        target_to_source_distances=target_to_source_distances.astype(np.float32),
        source_to_target_face_indices=source_to_target_faces.astype(np.int32),
        source_to_target_vertex_indices=target_face_vertex_indices.astype(np.int32),
        source_to_target_barycentric=source_to_target_barycentric.astype(np.float32),
        source_to_target_distances=source_to_target_distances.astype(np.float32),
        rotation_vector=to_numpy(fit.rotation_vector).astype(np.float32),
        scale_delta=np.array(float(torch.exp(fit.scale_delta).cpu()), dtype=np.float32),
        translation=to_numpy(fit.translation).astype(np.float32),
        loss_history_json=np.array(json.dumps([asdict(row) for row in fit.loss_history])),
        metadata_json=np.array(json.dumps(metadata, indent=2, sort_keys=True)),
        **source_fit,
    )


@app.command()
def main(
    source_model: Annotated[str, typer.Argument(help="Source model topology to deform.")],
    target_model: Annotated[str, typer.Argument(help="Target model to keep fixed.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Destination .npz mapping file.")] = Path(
        "mapping.npz"
    ),
    source_path: Annotated[Path | None, typer.Option(help="Optional source model path.")] = None,
    target_path: Annotated[Path | None, typer.Option(help="Optional target model path.")] = None,
    source_gender: Annotated[str, typer.Option(help="SMPL-family source gender.")] = "neutral",
    target_gender: Annotated[str, typer.Option(help="SMPL-family target gender.")] = "neutral",
    source_simplify: Annotated[float, typer.Option(help="Source mesh simplification factor; 1 keeps full mesh.")] = 1.0,
    target_simplify: Annotated[float, typer.Option(help="Target mesh simplification factor; 1 keeps full mesh.")] = 1.0,
    source_lod: Annotated[int, typer.Option(help="MHR source LOD.")] = 1,
    target_lod: Annotated[int, typer.Option(help="MHR target LOD.")] = 1,
    source_soma_type: Annotated[str, typer.Option(help="SOMA source model_type.")] = "soma",
    target_soma_type: Annotated[str, typer.Option(help="SOMA target model_type.")] = "soma",
    source_anny_rig: Annotated[str, typer.Option(help="ANNY source rig.")] = "default",
    target_anny_rig: Annotated[str, typer.Option(help="ANNY target rig.")] = "default",
    source_anny_topology: Annotated[str, typer.Option(help="ANNY source topology.")] = "default",
    target_anny_topology: Annotated[str, typer.Option(help="ANNY target topology.")] = "default",
    device: Annotated[str, typer.Option(help="Torch device: auto, cpu, cuda, or cuda:N.")] = "auto",
    seed: Annotated[int, typer.Option(help="Random seed for surface sampling.")] = 13,
    sample_points: Annotated[int, typer.Option(help="Surface samples per side for every optimization step.")] = 4096,
    fit_robust_scale: Annotated[
        float,
        typer.Option(help="Robust surface-loss scale as a fraction of the target bounding-box diagonal."),
    ] = 0.03,
    mesh_weight: Annotated[
        float,
        typer.Option(help="Weight for closest-triangle mesh surface loss added to sampled Chamfer."),
    ] = 1.0,
    mesh_point_plane_weight: Annotated[
        float,
        typer.Option(help="Point-to-plane weight inside the closest-triangle mesh surface loss."),
    ] = 0.2,
    mesh_query_every: Annotated[
        int,
        typer.Option(help="Optimization steps between closest-triangle correspondence refreshes."),
    ] = 10,
    pose_steps: Annotated[int, typer.Option(help="Source pose/shape fitting iterations before surface fitting.")] = 500,
    pose_lr: Annotated[float, typer.Option(help="Source pose fitting learning rate.")] = 2.0e-2,
    pose_regularizer: Annotated[
        float, typer.Option(help="Weight keeping fitted source parameters near rest.")
    ] = 3.0e-5,
    rigid_steps: Annotated[int, typer.Option(help="Rigid similarity fitting iterations.")] = 80,
    smooth_steps: Annotated[int, typer.Option(help="Strongly regularized non-rigid fitting iterations.")] = 180,
    detail_steps: Annotated[int, typer.Option(help="Final low-learning-rate refinement iterations.")] = 140,
    polish_steps: Annotated[int, typer.Option(help="Final regularization polish iterations.")] = 80,
    rigid_lr: Annotated[float, typer.Option(help="Rigid stage learning rate.")] = 2.0e-3,
    smooth_lr: Annotated[float, typer.Option(help="Smooth non-rigid stage learning rate.")] = 1.0e-3,
    detail_lr: Annotated[float, typer.Option(help="Detail non-rigid stage learning rate.")] = 5.0e-4,
    polish_lr: Annotated[float, typer.Option(help="Polish stage learning rate.")] = 2.0e-4,
    anchor_weight: Annotated[float, typer.Option(help="Rigid-stage weight for common skeleton joints.")] = 0.0,
    show: Annotated[bool, typer.Option("--show/--no-show", help="Show the fit live in viser.")] = True,
    viser_port: Annotated[int, typer.Option(help="Viser server port.")] = 8080,
    update_every: Annotated[int, typer.Option(help="Viewer update interval in optimization steps.")] = 25,
    metric_every: Annotated[int, typer.Option(help="Optimization metric/report interval in steps.")] = 50,
    verbose: Annotated[bool, typer.Option("--verbose", help="Print detailed model loading and stage logs.")] = False,
) -> None:
    configure_logging(verbose)

    if sample_points < 32:
        raise typer.BadParameter("sample_points must be at least 32.")
    if source_simplify < 1.0 or target_simplify < 1.0:
        raise typer.BadParameter("simplify factors must be >= 1.0.")
    if fit_robust_scale < 0.0:
        raise typer.BadParameter("fit_robust_scale must be non-negative.")
    if mesh_weight < 0.0:
        raise typer.BadParameter("mesh_weight must be non-negative.")
    if mesh_point_plane_weight < 0.0:
        raise typer.BadParameter("mesh_point_plane_weight must be non-negative.")
    if mesh_query_every < 1:
        raise typer.BadParameter("mesh_query_every must be at least 1.")
    if pose_steps < 0:
        raise typer.BadParameter("pose_steps must be non-negative.")
    if pose_lr <= 0.0:
        raise typer.BadParameter("pose_lr must be positive.")
    if min(rigid_steps, smooth_steps, detail_steps, polish_steps) < 0:
        raise typer.BadParameter("stage steps must be non-negative.")
    if min(rigid_lr, smooth_lr, detail_lr, polish_lr) <= 0.0:
        raise typer.BadParameter("stage learning rates must be positive.")
    if pose_regularizer < 0.0:
        raise typer.BadParameter("pose_regularizer must be non-negative.")
    if metric_every < 1:
        raise typer.BadParameter("metric_every must be at least 1.")
    if anchor_weight < 0.0:
        raise typer.BadParameter("anchor_weight must be non-negative.")

    started_at = time.perf_counter()
    torch_device = resolve_device(device)
    loguru.logger.info("Using torch device {}", torch_device)

    source = load_model(
        source_model,
        model_path=source_path,
        gender=source_gender,
        simplify=source_simplify,
        lod=source_lod,
        soma_model_type=source_soma_type,
        anny_rig=source_anny_rig,
        anny_topology=source_anny_topology,
        device=torch_device,
    )
    target = load_model(
        target_model,
        model_path=target_path,
        gender=target_gender,
        simplify=target_simplify,
        lod=target_lod,
        soma_model_type=target_soma_type,
        anny_rig=target_anny_rig,
        anny_topology=target_anny_topology,
        device=torch_device,
    )

    source, pose_history = pose_source_to_target(
        source,
        target,
        steps=pose_steps,
        lr=pose_lr,
        regularizer=pose_regularizer,
        metric_every=metric_every,
    )

    viewer = None
    if show:
        aligned_source, _, _, _ = initial_similarity(source.vertices, target.vertices)
        viewer = build_viewer(
            source_vertices=to_numpy(aligned_source),
            source_faces=source.faces,
            target_vertices=to_numpy(target.vertices),
            target_faces=target.faces,
            port=viser_port,
        )

    stages = [
        Stage("rigid", rigid_steps, rigid_lr, 0.0, 0.0, 0.0, 0.0),
        Stage("smooth", smooth_steps, smooth_lr, 0.025, 6.0, 0.005, 0.0),
        Stage("detail", detail_steps, detail_lr, 0.06, 12.0, 0.01, 0.01),
        Stage("polish", polish_steps, polish_lr, 0.2, 40.0, 0.05, 0.05),
    ]
    anchor_names, source_anchors, target_anchors = common_joint_anchors(source, target)
    loguru.logger.info("Using {} common skeleton anchors: {}", len(anchor_names), ", ".join(anchor_names) or "none")
    fit_problem = FitProblem(
        source_vertices=source.vertices,
        source_faces=source.faces,
        source_anchor_points=source_anchors,
        target_vertices=target.vertices,
        target_faces=target.faces,
        target_anchor_points=target_anchors,
    )
    fit_settings = FitSettings(
        stages=stages,
        sample_points=sample_points,
        fit_robust_scale=fit_robust_scale,
        mesh_weight=mesh_weight,
        mesh_point_plane_weight=mesh_point_plane_weight,
        mesh_query_every=mesh_query_every,
        anchor_weight=anchor_weight,
        viewer=viewer,
        seed=seed,
        update_every=update_every,
        metric_every=metric_every,
    )
    fit = fit_source_to_target(fit_problem, fit_settings)

    deformed_source_vertices = to_numpy(fit.vertices).astype(np.float32)
    target_vertices = to_numpy(target.vertices).astype(np.float32)
    loguru.logger.info("Projecting target vertices onto deformed source topology with Warp")
    target_to_source_faces, target_to_source_bary, target_to_source_dist = project_points_with_warp(
        mesh_vertices=deformed_source_vertices,
        mesh_faces=source.faces,
        query_points=target_vertices,
        torch_device=torch_device,
    )
    loguru.logger.info("Projecting deformed source vertices onto target topology with Warp")
    source_to_target_faces, source_to_target_bary, source_to_target_dist = project_points_with_warp(
        mesh_vertices=target_vertices,
        mesh_faces=target.faces,
        query_points=deformed_source_vertices,
        torch_device=torch_device,
    )

    bbox = target_vertices.max(axis=0) - target_vertices.min(axis=0)
    target_scale = float(np.linalg.norm(bbox))
    final_fit = fit.loss_history[-1]
    quality = MappingQuality(
        target_to_source=distance_metrics(target_to_source_dist, target_scale),
        source_to_target=distance_metrics(source_to_target_dist, target_scale),
        final_anchor_p95=final_fit.anchor_p95,
        final_edge_stretch_p95=final_fit.deformation.edge_stretch_p95,
        final_offset_p95=final_fit.deformation.offset_p95,
        final_flip_fraction=final_fit.flip_fraction,
    )
    warnings = quality_warnings(quality, target_scale, len(anchor_names))

    elapsed_seconds = time.perf_counter() - started_at
    metadata: dict[str, JsonValue] = {
        "source_model": source.model_name,
        "target_model": target.model_name,
        "source_path": str(source_path) if source_path else None,
        "target_path": str(target_path) if target_path else None,
        "source_vertex_count": int(source.vertices.shape[0]),
        "target_vertex_count": int(target.vertices.shape[0]),
        "source_triangle_count": int(source.faces.shape[0]),
        "target_triangle_count": int(target.faces.shape[0]),
        "sample_points": sample_points,
        "fit_robust_scale": fit_robust_scale,
        "mesh_weight": mesh_weight,
        "mesh_point_plane_weight": mesh_point_plane_weight,
        "mesh_query_every": mesh_query_every,
        "source_fit_steps": pose_steps,
        "source_fit_lr": pose_lr,
        "source_fit_regularizer": pose_regularizer,
        "source_fit_parameter_names": [name for name in SOURCE_FIT_PARAMETER_NAMES if name in source.params],
        "source_fit_history": [asdict(row) for row in pose_history],
        "fit_stages": [asdict(stage) for stage in stages],
        "initial_method": fit.initial_method,
        "initial_scale": fit.initial_scale,
        "anchor_weight": anchor_weight,
        "anchor_loss_stages": ["rigid"],
        "anchor_joint_count": len(anchor_names),
        "anchor_joint_names": anchor_names,
        "seed": seed,
        "quality": asdict(quality),
        "quality_warnings": warnings,
        "optimization_history": [asdict(row) for row in fit.loss_history],
        "elapsed_seconds": elapsed_seconds,
        "mapping_semantics": {
            "target_to_source": "target_vertices[i] ~= barycentric[i] over deformed_source_vertices[source_faces[face_indices[i]]]",
            "source_to_target": "deformed_source_vertices[i] ~= barycentric[i] over target_vertices[target_faces[face_indices[i]]]",
        },
    }
    save_mapping(
        output=output,
        source=source,
        target=target,
        fit=fit,
        target_to_source_faces=target_to_source_faces,
        target_to_source_barycentric=target_to_source_bary,
        target_to_source_distances=target_to_source_dist,
        source_to_target_faces=source_to_target_faces,
        source_to_target_barycentric=source_to_target_bary,
        source_to_target_distances=source_to_target_dist,
        metadata=metadata,
    )
    print_summary(quality, warnings, fit.loss_history, pose_history, output, elapsed_seconds)

    loguru.logger.info(
        "Saved {} (target->source p95 {:.6e}, source->target p95 {:.6e})",
        output,
        quality.target_to_source.p95,
        quality.source_to_target.p95,
    )
    loguru.logger.info("Finished in {:.1f}s", elapsed_seconds)


if __name__ == "__main__":
    app()
