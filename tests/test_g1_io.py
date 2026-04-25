from pathlib import Path

import pytest

from body_models.g1 import io

pytestmark = pytest.mark.fast


def test_g1_download_model_uses_huggingface_layout(monkeypatch, tmp_path: Path) -> None:
    urls: list[str] = []

    def fake_urlretrieve(url: str, filename: Path):
        urls.append(url)
        filename.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith(".xml"):
            filename.write_text("<mujoco/>")
        else:
            filename.write_text(f"solid {filename.stem}\nendsolid {filename.stem}\n")
        return filename, None

    monkeypatch.setattr(io, "get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(io.urllib.request, "urlretrieve", fake_urlretrieve)

    path = io.download_model()

    assert path == tmp_path / "g1"
    assert (path / "xml" / "g1.xml").exists()
    for mesh_name in {mesh for meshes in io.G1_MESH_JOINT_MAP.values() for mesh in meshes}:
        assert (path / "meshes" / "g1" / mesh_name).exists()
    assert urls[0] == f"{io.G1_HF_BASE_URL}/{io.G1_HF_XML}"
    assert all(url.startswith(f"{io.G1_HF_BASE_URL}/") for url in urls)
