#!/usr/bin/env python3
"""Gradio viewer for D4RT artifacts (browse precomputed results; no GPU re-run).

Tab "Demo Packages": pick a ``build_demo_from_video.py`` output directory under
``--results-root`` and inspect its input video, poster, ``meta`` table, and a GLB
point cloud (rebuildable with different filters via "Update Visual").

Tab "Trajectory Check": point at a static-tracks ``.npz`` and a COLMAP model to
compute the camera-trajectory consistency metrics and export a rerun ``.rrd``.

Security: this is a local dev tool. The Trajectory Check tab reads whatever
filesystem paths you type with the running user's permissions, so it defaults to
binding ``127.0.0.1`` only. Pass ``--server-name 0.0.0.0`` to expose it on the
network, and only do so on a trusted network.

Run::

    uv run --extra vis python scripts/demo_gradio.py --results-root tmp/
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import gradio as gr

# Local script execution needs repo root on sys.path before project imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._gradio_helpers import (  # noqa: E402
    build_glb_from_demo_data,
    discover_demo_packages,
    meta_table_rows,
)
from vis.rerun_visualize import save_trajectory_comparison_to_rrd  # noqa: E402

_EMPTY_PKG = (None, None, [], None)
# Generated GLBs go to a temp dir, never into the (possibly read-only / shared)
# source package directory.
_GLB_CACHE_DIR = Path(tempfile.gettempdir()) / "d4rt_gradio_glb"


def _load_package(pkg: str | None, show_dynamic: bool, max_points: float) -> tuple:
    """Return (video, poster, meta-rows, glb) for the selected demo package."""
    if not pkg:
        return _EMPTY_PKG
    pkg_dir = Path(pkg)
    try:
        data = json.loads((pkg_dir / "assets" / "demo_data.json").read_text())
        manifest = json.loads((pkg_dir / "manifest.json").read_text())
    except (OSError, ValueError) as exc:
        return None, None, [["error", f"failed to read package: {exc}"]], None
    video = pkg_dir / manifest.get("video_copy", "assets/input_video.mp4")
    poster = pkg_dir / manifest.get("video_poster", "assets/video_poster.jpg")
    _GLB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    glb_name = f"{pkg_dir.name}_{int(bool(show_dynamic))}_{int(max_points)}.glb"
    glb = build_glb_from_demo_data(
        data,
        _GLB_CACHE_DIR / glb_name,
        show_dynamic=bool(show_dynamic),
        max_points=int(max_points),
    )
    return (
        str(video) if video.exists() else None,
        str(poster) if poster.exists() else None,
        meta_table_rows(data["meta"]),
        str(glb),
    )


def _check_trajectory(npz: str, colmap: str) -> tuple[list[list[str]], str | None]:
    """Run the COLMAP trajectory-consistency check and export a rerun .rrd."""
    from scripts.check_colmap_trajectory_consistency import (  # noqa: PLC0415
        compute_consistency,
        load_prediction,
        read_colmap_model,
    )

    if not npz or not colmap:
        return [["error", "provide both an .npz path and a COLMAP model dir"]], None
    report = compute_consistency(
        load_prediction(Path(npz)), read_colmap_model(Path(colmap)), use_ransac=True
    )
    rows = [
        [k, json.dumps(v) if isinstance(v, dict) else str(v)]
        for k, v in report.items()
        if not k.startswith("_")
    ]
    rrd = save_trajectory_comparison_to_rrd(
        Path(npz), Path(colmap), REPO_ROOT / "outputs" / "gradio_traj.rrd"
    )
    return rows, str(rrd)


def build_ui(results_root: str | Path) -> gr.Blocks:
    """Construct (but do not launch) the gradio Blocks app."""
    results_root = Path(results_root)
    choices = [str(p) for p in discover_demo_packages(results_root)]

    with gr.Blocks(title="D4RT Visualizer") as demo:
        gr.Markdown(
            "# D4RT result viewer\nBrowse precomputed demo packages and trajectory checks."
        )

        with gr.Tab("Demo Packages"):
            with gr.Row():
                with gr.Column(scale=2):
                    pkg_dd = gr.Dropdown(
                        choices=choices,
                        value=choices[0] if choices else None,
                        label="Demo package",
                    )
                    refresh_btn = gr.Button("Refresh list")
                    video = gr.Video(label="Input video", interactive=False)
                    poster = gr.Image(label="Poster", interactive=False)
                    meta_df = gr.Dataframe(headers=["key", "value"], label="meta")
                with gr.Column(scale=3):
                    show_dyn = gr.Checkbox(label="Highlight dynamic points", value=True)
                    max_pts = gr.Slider(
                        10_000, 500_000, value=200_000, step=10_000, label="Max points"
                    )
                    model3d = gr.Model3D(label="Point cloud (GLB)", height=600)
                    update_btn = gr.Button("Update Visual", variant="primary")

            outputs = [video, poster, meta_df, model3d]
            pkg_dd.change(_load_package, [pkg_dd, show_dyn, max_pts], outputs)
            update_btn.click(_load_package, [pkg_dd, show_dyn, max_pts], outputs)
            refresh_btn.click(
                lambda: gr.update(
                    choices=[str(p) for p in discover_demo_packages(results_root)]
                ),
                outputs=[pkg_dd],
            )

        with gr.Tab("Trajectory Check"):
            npz_in = gr.Textbox(label="static-tracks .npz path")
            colmap_in = gr.Textbox(label="COLMAP model dir")
            run_btn = gr.Button("Compute consistency", variant="primary")
            metrics_df = gr.Dataframe(headers=["metric", "value"], label="metrics")
            rrd_file = gr.File(label="rerun .rrd (open with: uv run rerun <file>)")
            run_btn.click(
                _check_trajectory, [npz_in, colmap_in], [metrics_df, rrd_file]
            )

    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description="D4RT gradio viewer")
    parser.add_argument(
        "--results-root", required=True, help="Root dir to scan for demo packages."
    )
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="Bind address. Localhost by default; use 0.0.0.0 only on trusted networks.",
    )
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_ui(args.results_root).queue().launch(
        server_name=args.server_name, server_port=args.server_port, share=args.share
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
