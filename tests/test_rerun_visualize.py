from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_demo_from_video as demo
from tests.test_build_demo_from_video import _make_gradient_video, make_demo_package
from tests.test_check_colmap_trajectory_consistency import (
    _build_synthetic_scene,
    _write_colmap_text,
    _write_pred_npz,
)
from vis import rerun_visualize as rv


def _make_demo_package_dir(tmp_path: Path) -> Path:
    package = make_demo_package()
    video = _make_gradient_video(3, 16, 20)
    out = tmp_path / "demo_pkg"
    src = tmp_path / "src.gif"
    src.write_bytes(b"x")
    demo._write_demo_package(
        output_dir=out,
        input_path=src,
        package=cast(demo.DemoPackage, package),
        video_rgb=video,
        fps=6.0,
    )
    return out


def test_save_demo_package_to_rrd_writes_nonempty_file(tmp_path: Path) -> None:
    pkg_dir = _make_demo_package_dir(tmp_path)
    rrd = tmp_path / "demo.rrd"
    out = rv.save_demo_package_to_rrd(pkg_dir, rrd)
    assert out == rrd
    assert rrd.stat().st_size > 1024


def test_save_demo_package_to_rrd_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rv.save_demo_package_to_rrd(tmp_path / "nope", tmp_path / "x.rrd")


def test_visualize_static_tracks_rejects_missing_file(tmp_path: Path) -> None:
    # The spawn-mode entrypoint must fail fast (no viewer launched) on bad input.
    with pytest.raises(FileNotFoundError):
        rv.visualize_static_tracks(tmp_path / "missing.npz")


def test_save_static_tracks_to_rrd_writes_nonempty_file(tmp_path: Path) -> None:
    k, pts, uv, names, _, _, _, _, _ = _build_synthetic_scene()
    npz = tmp_path / "pred.npz"
    _write_pred_npz(npz, k, pts, uv, names)
    rrd = tmp_path / "tracks.rrd"
    out = rv.save_static_tracks_to_rrd(npz, rrd)
    assert out == rrd
    assert rrd.stat().st_size > 1024


def test_save_trajectory_comparison_to_rrd_writes_nonempty_file(tmp_path: Path) -> None:
    k, pts, uv, names, _, _, _, colmap_r, colmap_t = _build_synthetic_scene()
    model = tmp_path / "model"
    _write_colmap_text(model, k, names, colmap_r, colmap_t)
    npz = tmp_path / "pred.npz"
    _write_pred_npz(npz, k, pts, uv, names)
    rrd = tmp_path / "traj.rrd"
    out = rv.save_trajectory_comparison_to_rrd(npz, model, rrd)
    assert out == rrd
    assert rrd.stat().st_size > 1024
