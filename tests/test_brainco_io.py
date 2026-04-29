from pathlib import Path

import pytest

from body_models.brainco import io

pytestmark = pytest.mark.fast


def test_brainco_download_extracts_official_layout(monkeypatch, tmp_path: Path) -> None:
    import io as bytes_io
    import zipfile

    archive = bytes_io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Revo2_xml/xml_left/brainco-lefthand-v2.xml", "<mujoco><worldbody/></mujoco>")
        zf.writestr("Revo2_xml/xml_left/meshes/left_base_link.STL", "solid l\nendsolid l\n")
        zf.writestr("Revo2_xml/xml_right/brainco-righthand-v2.xml", "<mujoco><worldbody/></mujoco>")
        zf.writestr("Revo2_xml/xml_right/meshes/base_link.STL", "solid r\nendsolid r\n")
    archive_bytes = archive.getvalue()

    def fake_urlretrieve(url: str, filename: Path):
        assert url == io.BRAINCO_REVO2_MUJOCO_URL
        filename.write_bytes(archive_bytes)
        return filename, None

    monkeypatch.setattr(io, "get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(io.urllib.request, "urlretrieve", fake_urlretrieve)

    path = io.download_model()

    assert path == tmp_path / "brainco"
    assert (path / "left.xml").exists()
    assert (path / "right.xml").exists()
    assert (path / "meshes" / "left" / "left_base_link.STL").exists()
    assert (path / "meshes" / "right" / "right_base_link.STL").exists()


def test_brainco_official_assets_load_from_cache() -> None:
    path = io.get_cache_dir() / "brainco"
    if not (path / "right.xml").exists():
        pytest.skip(f"BrainCo test assets not found at {path}")
    data = io.load_model_data(path, side="right")

    assert len(data["joint_names"]) == 12
    assert len(data["parents"]) == 12
    assert len(data["qpos_joint_names"]) == 6
    assert data["joint_names"][0] == "right_base_skel"
    assert data["qpos_joint_names"] == [
        "right_thumb_metacarpal_skel",
        "right_thumb_proximal_skel",
        "right_index_proximal_skel",
        "right_middle_proximal_skel",
        "right_ring_proximal_skel",
        "right_pinky_proximal_skel",
    ]
    assert data["coupled_joint_indices"] == [5, 7, 9, 11, 3]
    assert data["coupled_driver_indices"] == [2, 3, 4, 5, 1]
    assert data["vertices"].shape[1] == 3
    assert data["faces"].shape[1] == 3
