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

ASSET_DIR = Path(__file__).parent / "assets"
RTOL = 1e-4
ATOL = 1e-4
SEEDS = {"soma": 101, "mhr": 202, "smplx": 303}


@pytest.fixture(scope="module")
def upstream_data_root() -> Path:
    pytest.importorskip("soma")

    from soma.assets import get_assets_dir

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return Path(get_assets_dir())


def _make_upstream_layer(model_type: str, upstream_data_root: Path):
    pytest.importorskip("soma")

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
    # Our native SOMA implementation matches the upstream PyTorch rotation fitter.
    # Keep the oracle on that path so these tests isolate identity-backend parity.
    layer.skeleton_transfer.use_warp_for_rotations = False
    return layer


def _sample_inputs(
    num_identity_coeffs: int,
    num_scale_params: int | None,
    *,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    inputs = {
        "pose": torch.tensor(rng.standard_normal((1, 77, 3)).astype(np.float32) * 0.05),
        "identity": torch.tensor(rng.standard_normal((1, num_identity_coeffs)).astype(np.float32) * 0.1),
        "global_translation": torch.tensor(rng.standard_normal((1, 3)).astype(np.float32) * 0.01),
    }
    if num_scale_params is not None:
        inputs["scale_params"] = torch.tensor(rng.standard_normal((1, num_scale_params)).astype(np.float32) * 0.05)
    return inputs


def _upstream_forward(layer, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    scale_params = inputs.get("scale_params")
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        return layer(
            inputs["pose"],
            inputs["identity"],
            scale_params=scale_params,
            transl=inputs["global_translation"],
            pose2rot=True,
        )


def _joint_positions(transforms: torch.Tensor) -> torch.Tensor:
    return transforms[..., :3, 3]


@pytest.fixture(scope="module")
def soma_model_path() -> Path:
    return get_model_path()


@pytest.mark.parametrize("model_type", ["soma", "mhr", "smplx"])
def test_torch_identity_backends_match_upstream(
    model_type: str,
    soma_model_path: Path,
    upstream_data_root: Path,
) -> None:
    from body_models.soma.torch import SOMA

    upstream = _make_upstream_layer(model_type, upstream_data_root)
    inputs = _sample_inputs(
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

    torch.testing.assert_close(verts, reference["vertices"], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(_joint_positions(transforms), reference["joints"], rtol=RTOL, atol=ATOL)
