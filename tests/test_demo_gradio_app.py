from __future__ import annotations

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
