"""GarmentMeasurements I/O tests that do not require private model assets."""

from pathlib import Path

import numpy as np

from body_models.garment_measurements import io


def test_upstream_folder_is_preprocessed_to_platform_cache(tmp_path, monkeypatch) -> None:
    upstream = tmp_path / "GarmentMeasurements" / "data"
    (upstream / "pca").mkdir(parents=True)
    (upstream / "template").mkdir()
    (upstream / "pca" / "point.pca").write_bytes(b"pca")
    (upstream / "pca" / "mean.obj").write_text("v 0 0 0\n")
    (upstream / "template" / "male.fbx").write_bytes(b"fbx")

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(io, "get_cache_dir", lambda: cache_dir)

    generated: list[tuple[Path, Path]] = []

    def fake_run_asset_generator(upstream_data: Path, output_dir: Path) -> None:
        generated.append((upstream_data, output_dir))
        _write_minimal_preprocessed_model(output_dir / io.PREPROCESSED_FILENAME)

    monkeypatch.setattr(io, "_run_asset_generator", fake_run_asset_generator)

    data = io.load_model_data(upstream)

    assert generated == [(upstream, io._preprocessed_output_dir(upstream))]
    assert (generated[0][1] / io.PREPROCESSED_FILENAME).is_file()
    assert data["mean_vertices"].shape == (3, 3)
    assert data["skin_weights"].shape == (3, 1)


def test_run_asset_generator_uses_uv_pep723_script(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(io.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(io.subprocess, "run", lambda cmd, check: calls.append((cmd, check)))

    io._run_asset_generator(tmp_path / "data", tmp_path / "out")

    cmd, check = calls[0]
    assert check is True
    assert cmd[:5] == ["/usr/bin/uv", "run", "--python", io.GENERATOR_PYTHON, "--no-project"]
    assert cmd[-2:] == [str(tmp_path / "data"), str(tmp_path / "out")]
    assert Path(cmd[-3]).name == "generate_asset.py"


def test_run_asset_generator_fails_clearly_without_uv(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(io.shutil, "which", lambda name: None)

    try:
        io._run_asset_generator(tmp_path / "data", tmp_path / "out")
    except RuntimeError as exc:
        assert "requires `uv` on PATH" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def _write_minimal_preprocessed_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean_vertices=np.zeros((3, 3), dtype=np.float32),
        components=np.zeros((3, 3, 1), dtype=np.float32),
        eigenvalues=np.ones((1,), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        joint_names=np.asarray(["root"]),
        parents=np.array([-1], dtype=np.int64),
        bind_quats=np.array([[1, 0, 0, 0]], dtype=np.float32),
        skin_weights=np.ones((3, 1), dtype=np.float32),
        mvc_weights=np.ones((3, 1), dtype=np.float32) / 3.0,
    )
