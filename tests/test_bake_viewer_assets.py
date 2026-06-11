from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import bake_viewer_assets as bake
from scripts import build_demo_from_video as demo
from tests.test_build_demo_from_video import _make_gradient_video, make_demo_package
from tests.test_check_colmap_trajectory_consistency import (
    _build_synthetic_scene,
    _write_colmap_text,
    _write_pred_npz,
)


def _make_pkg(root: Path, name: str) -> Path:
    pkg = root / name
    src = root / f"{name}.gif"
    src.write_bytes(b"x")
    demo._write_demo_package(
        output_dir=pkg,
        input_path=src,
        package=cast(demo.DemoPackage, make_demo_package()),
        video_rgb=_make_gradient_video(3, 16, 20),
        fps=6.0,
    )
    return pkg


def test_bake_demo_package_writes_glb_and_assets(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    _make_pkg(root, "pkgA")
    out = tmp_path / "baked"

    entry = bake.bake_demo_package(root / "pkgA", out)

    assert entry["kind"] == "demo_package"
    assert (out / entry["glb"]).stat().st_size > 0
    assert (out / entry["video"]).exists()
    assert (out / entry["poster"]).exists()
    assert "numFrames" in {r[0] for r in entry["meta_rows"]}


def test_bake_trajectory_writes_report_plot_rrd(tmp_path: Path) -> None:
    k, pts, uv, names, _, _, _, colmap_r, colmap_t = _build_synthetic_scene()
    model = tmp_path / "model"
    _write_colmap_text(model, k, names, colmap_r, colmap_t)
    npz = tmp_path / "pred.npz"
    _write_pred_npz(npz, k, pts, uv, names)
    out = tmp_path / "baked"

    entry = bake.bake_trajectory("t1", npz, model, out)

    assert (out / entry["report"]).exists()
    assert (out / entry["plot"]).stat().st_size > 0
    assert (out / entry["rrd"]).stat().st_size > 0
    assert any(r[0] == "ate_rmse" for r in entry["metrics_rows"])


def test_bake_all_writes_index(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    _make_pkg(root, "pkgA")
    _make_pkg(root, "pkgB")
    k, pts, uv, names, _, _, _, colmap_r, colmap_t = _build_synthetic_scene()
    model = tmp_path / "model"
    _write_colmap_text(model, k, names, colmap_r, colmap_t)
    npz = tmp_path / "pred.npz"
    _write_pred_npz(npz, k, pts, uv, names)
    out = tmp_path / "baked"

    index_path = bake.bake_all(root, out, trajectories=(("t1", str(npz), str(model)),))

    assert index_path == out / "viewer_index.json"
    index = json.loads(index_path.read_text())
    assert len(index["demo_packages"]) == 2
    assert len(index["trajectories"]) == 1
    # Every referenced asset actually exists (self-contained bake).
    for entry in index["demo_packages"]:
        assert (out / entry["glb"]).exists()
    assert (out / index["trajectories"][0]["plot"]).exists()
