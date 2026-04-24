"""Upstream parity tests for SOMA identity backends.

These tests use the official ``py-soma-x`` package as the reference
implementation for the torch backend.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest
import torch

from body_models.soma.io import get_model_path

pytestmark = pytest.mark.fast

ASSET_DIR = Path(__file__).parent / "assets"
RTOL = 1e-4
ATOL = 1e-4
SEEDS = {"soma": 101, "anny": 151, "mhr": 202, "smplx": 303}
MODEL_RTOL = {"mhr": 2e-4}
MODEL_ATOL = {"mhr": 2e-4}


@pytest.fixture(scope="module")
def upstream_data_root() -> Path:
    pytest.importorskip("soma")

    from soma.assets import get_assets_dir

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return Path(get_assets_dir())


def _make_upstream_layer(model_type: str, upstream_data_root: Path):
    pytest.importorskip("soma")
    if model_type == "anny":
        pytest.importorskip("anny")

    from soma import SOMALayer

    identity_model_kwargs: dict[str, str] | None = None
    if model_type == "smplx":
        identity_model_kwargs = {"model_path": str((ASSET_DIR / "smplx" / "model" / "SMPLX_NEUTRAL.npz").resolve())}

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        layer = SOMALayer(
            data_root=upstream_data_root,
            device="cpu",
            identity_model_type=model_type,
            identity_model_kwargs=identity_model_kwargs,
        )
    return layer


def _sample_inputs(
    model_type: str,
    num_identity_coeffs: int,
    num_scale_params: int | None,
    *,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    if model_type == "anny":
        identity = torch.tensor(rng.uniform(0.2, 0.8, size=(1, num_identity_coeffs)).astype(np.float32))
    else:
        identity = torch.tensor(rng.standard_normal((1, num_identity_coeffs)).astype(np.float32) * 0.1)
    inputs = {
        "pose": torch.tensor(rng.standard_normal((1, 77, 3)).astype(np.float32) * 0.05),
        "identity": identity,
        "global_translation": torch.tensor(rng.standard_normal((1, 3)).astype(np.float32) * 0.01),
    }
    if num_scale_params is not None:
        inputs["scale_params"] = torch.tensor(rng.standard_normal((1, num_scale_params)).astype(np.float32) * 0.05)
    return inputs


def _upstream_forward(layer, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    scale_params = inputs.get("scale_params")
    identity = inputs["identity"]
    if layer.identity_model_type == "anny":
        labels = tuple(layer.identity_model.identity_model.phenotype_labels)
        identity = {label: identity[:, i] for i, label in enumerate(labels)}
        scale_params = {
            label: torch.zeros_like(inputs["identity"][:, 0])
            for label in layer.identity_model.identity_model.local_change_labels
        }
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        return layer(
            inputs["pose"],
            identity,
            scale_params=scale_params,
            transl=inputs["global_translation"],
            pose2rot=True,
        )


def _joint_positions(transforms: torch.Tensor) -> torch.Tensor:
    return transforms[..., :3, 3]


@pytest.fixture(scope="module")
def soma_model_path() -> Path:
    return get_model_path()


@pytest.mark.parametrize("model_type", ["soma", "anny", "mhr", "smplx"])
def test_torch_identity_backends_match_upstream(
    model_type: str,
    soma_model_path: Path,
    upstream_data_root: Path,
) -> None:
    from body_models.soma.torch import SOMA

    upstream = _make_upstream_layer(model_type, upstream_data_root)
    inputs = _sample_inputs(
        model_type,
        upstream.identity_model.num_identity_coeffs,
        upstream.identity_model.num_scale_params,
        seed=SEEDS[model_type],
    )
    reference = _upstream_forward(upstream, inputs)

    model = SOMA(model_path=soma_model_path, model_type=model_type)

    with torch.no_grad():
        verts = model.forward_vertices(
            pose=inputs["pose"],
            global_translation=inputs["global_translation"],
            identity=inputs["identity"],
            scale_params=inputs.get("scale_params"),
        )
        transforms = model.forward_skeleton(
            pose=inputs["pose"],
            global_translation=inputs["global_translation"],
            identity=inputs["identity"],
            scale_params=inputs.get("scale_params"),
        )

    rtol = MODEL_RTOL.get(model_type, RTOL)
    atol = MODEL_ATOL.get(model_type, ATOL)
    torch.testing.assert_close(verts, reference["vertices"], rtol=rtol, atol=atol)
    torch.testing.assert_close(_joint_positions(transforms), reference["joints"], rtol=rtol, atol=atol)


def test_torch_identity_backend_can_match_upstream_torch_rotation_fitter(
    soma_model_path: Path,
    upstream_data_root: Path,
) -> None:
    from body_models.soma.torch import SOMA

    upstream = _make_upstream_layer("mhr", upstream_data_root)
    upstream.skeleton_transfer.use_warp_for_rotations = False
    inputs = _sample_inputs(
        "mhr",
        upstream.identity_model.num_identity_coeffs,
        upstream.identity_model.num_scale_params,
        seed=SEEDS["mhr"],
    )
    reference = _upstream_forward(upstream, inputs)

    model = SOMA(
        model_path=soma_model_path,
        model_type="mhr",
        match_warp=False,
    )

    with torch.no_grad():
        verts = model.forward_vertices(
            pose=inputs["pose"],
            global_translation=inputs["global_translation"],
            identity=inputs["identity"],
            scale_params=inputs.get("scale_params"),
        )
        transforms = model.forward_skeleton(
            pose=inputs["pose"],
            global_translation=inputs["global_translation"],
            identity=inputs["identity"],
            scale_params=inputs.get("scale_params"),
        )

    torch.testing.assert_close(verts, reference["vertices"], rtol=MODEL_RTOL["mhr"], atol=MODEL_ATOL["mhr"])
    torch.testing.assert_close(
        _joint_positions(transforms),
        reference["joints"],
        rtol=MODEL_RTOL["mhr"],
        atol=MODEL_ATOL["mhr"],
    )
