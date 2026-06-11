from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_demo_from_video as demo
from scripts import demo_gradio
from tests.test_build_demo_from_video import _make_gradient_video, make_demo_package


def test_build_ui_constructs_blocks_on_empty_root(tmp_path: Path) -> None:
    ui = demo_gradio.build_ui(tmp_path)
    assert isinstance(ui, gr.Blocks)


def test_load_package_returns_assets_and_glb(tmp_path: Path) -> None:
    package = make_demo_package()
    video = _make_gradient_video(3, 16, 20)
    pkg = tmp_path / "pkg"
    src = tmp_path / "s.gif"
    src.write_bytes(b"x")
    demo._write_demo_package(
        output_dir=pkg,
        input_path=src,
        package=cast(demo.DemoPackage, package),
        video_rgb=video,
        fps=6.0,
    )

    video_out, poster_out, rows, glb_out = demo_gradio._load_package(
        str(pkg), show_dynamic=True, max_points=200_000
    )

    assert video_out is not None and Path(video_out).exists()
    assert poster_out is not None and Path(poster_out).exists()
    assert "numFrames" in {r[0] for r in rows}
    assert glb_out is not None and Path(glb_out).stat().st_size > 0


def test_load_package_empty_on_no_selection() -> None:
    assert demo_gradio._load_package(None, show_dynamic=True, max_points=1000) == (
        None,
        None,
        [],
        None,
    )


def test_load_baked_package_resolves_paths(tmp_path: Path) -> None:
    entry = {
        "video": "a/input_video.mp4",
        "poster": None,
        "meta_rows": [["numFrames", "3"]],
        "glb": "a/scene.glb",
    }
    video, poster, rows, glb = demo_gradio.load_baked_package(tmp_path, entry)
    assert video == str(tmp_path / "a/input_video.mp4")
    assert poster is None
    assert rows == [["numFrames", "3"]]
    assert glb == str(tmp_path / "a/scene.glb")


def test_load_baked_trajectory_resolves_paths(tmp_path: Path) -> None:
    entry = {
        "metrics_rows": [["ate_rmse", "0.1"]],
        "plot": "t/traj.png",
        "rrd": "t/traj.rrd",
    }
    rows, plot, rrd = demo_gradio.load_baked_trajectory(tmp_path, entry)
    assert rows == [["ate_rmse", "0.1"]]
    assert plot == str(tmp_path / "t/traj.png")
    assert rrd == str(tmp_path / "t/traj.rrd")


def test_build_prebaked_ui_constructs_blocks(tmp_path: Path) -> None:
    (tmp_path / "viewer_index.json").write_text(
        json.dumps(
            {
                "demo_packages": [
                    {
                        "name": "a",
                        "glb": "a/scene.glb",
                        "meta_rows": [["numFrames", "3"]],
                        "video": None,
                        "poster": None,
                    }
                ],
                "trajectories": [
                    {
                        "name": "t",
                        "metrics_rows": [["ate_rmse", "0.1"]],
                        "plot": "t/traj.png",
                        "rrd": "t/traj.rrd",
                    }
                ],
            }
        )
    )
    ui = demo_gradio.build_prebaked_ui(tmp_path)
    assert isinstance(ui, gr.Blocks)
