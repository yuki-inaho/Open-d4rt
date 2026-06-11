from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import _gradio_helpers as gh
from scripts import build_demo_from_video as demo
from tests.test_build_demo_from_video import _make_gradient_video, make_demo_package


def _make_demo_data(tmp_path: Path) -> tuple[Path, dict]:
    package = make_demo_package()
    video = _make_gradient_video(3, 16, 20)
    out = tmp_path / "pkg"
    src = tmp_path / "s.gif"
    src.write_bytes(b"x")
    demo._write_demo_package(
        output_dir=out,
        input_path=src,
        package=cast(demo.DemoPackage, package),
        video_rgb=video,
        fps=6.0,
    )
    data = json.loads((out / "assets" / "demo_data.json").read_text())
    return out, data


def test_build_glb_from_demo_data_creates_loadable_file(tmp_path: Path) -> None:
    _, data = _make_demo_data(tmp_path)
    glb = tmp_path / "scene.glb"
    out = gh.build_glb_from_demo_data(data, glb)
    assert out == glb
    assert glb.stat().st_size > 0
    loaded = trimesh.load(str(glb))
    geoms = loaded.geometry if hasattr(loaded, "geometry") else {"_": loaded}
    assert len(geoms) >= 1


def test_build_glb_rejects_out_of_range_frame(tmp_path: Path) -> None:
    _, data = _make_demo_data(tmp_path)
    with pytest.raises(ValueError, match="out of range"):
        gh.build_glb_from_demo_data(data, tmp_path / "x.glb", frame=999)


def test_discover_demo_packages_finds_manifests(tmp_path: Path) -> None:
    for name in ("a", "b"):
        (tmp_path / name / "assets").mkdir(parents=True)
        (tmp_path / name / "manifest.json").write_text("{}")
    (tmp_path / "c").mkdir()
    found = gh.discover_demo_packages(tmp_path)
    assert {p.name for p in found} == {"a", "b"}


def test_discover_demo_packages_handles_missing_root(tmp_path: Path) -> None:
    assert gh.discover_demo_packages(tmp_path / "nope") == []


def test_meta_table_rows_format(tmp_path: Path) -> None:
    _, data = _make_demo_data(tmp_path)
    rows = gh.meta_table_rows(data["meta"])
    assert all(len(r) == 2 for r in rows)
    assert "numFrames" in {r[0] for r in rows}
