#!/usr/bin/env python3
"""Visualize D4RT artifacts with Rerun.

Inputs (mutually exclusive, one required):

* ``--demo-package DIR``  — a ``build_demo_from_video.py`` output directory.
* ``--tracks-npz FILE``   — a ``dump_static_tracks_for_trajectory.py`` dump.
  Add ``--colmap-model DIR`` to overlay the COLMAP trajectory comparison.

Modes:

* ``viewer``     — spawn the local Rerun viewer (needs a display).
* ``rrd``        — write a ``.rrd`` file only (headless; default).
* ``screenshot`` — write the ``.rrd``, serve it with ``rerun --serve-web`` and
  capture a PNG of the web viewer with the Playwright CLI (headless boxes).

Example::

    uv run --extra vis python scripts/visualize_rerun.py \\
        --tracks-npz preds.npz --colmap-model sparse_txt \\
        --mode screenshot --output out/traj.rrd --screenshot out/traj.png
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import TextIO

# Local script execution needs repo root on sys.path before project imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vis.rerun_visualize import (  # noqa: E402
    save_demo_package_to_rrd,
    save_dense_scene_to_rrd,
    save_static_tracks_to_rrd,
    save_trajectory_comparison_to_rrd,
    visualize_demo_package,
    visualize_dense_scene,
    visualize_static_tracks,
    visualize_trajectory_comparison,
)

SCREENSHOT_VIEWPORT = "1600,900"
SCREENSHOT_TIMEOUT_MS = "60000"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo-package", type=Path, help="Demo package directory.")
    source.add_argument("--tracks-npz", type=Path, help="Static-tracks .npz dump.")
    source.add_argument(
        "--dense-scene", type=Path, help="Dense-scene .npz (RGB/depth/points/camera)."
    )
    parser.add_argument(
        "--colmap-model", type=Path, help="COLMAP model dir (with --tracks-npz)."
    )
    parser.add_argument(
        "--mode", choices=("viewer", "rrd", "screenshot"), default="rrd"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/scene.rrd"))
    parser.add_argument("--screenshot", type=Path, default=Path("outputs/scene.png"))
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--no-ransac", action="store_true", help="Disable RANSAC PnP.")
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--grpc-port", type=int, default=9876)
    parser.add_argument("--serve-timeout", type=float, default=30.0)
    parser.add_argument("--render-wait", type=float, default=4.0)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Recording: dispatch to the right visualizer for the input artifact.
# ---------------------------------------------------------------------------


def _write_rrd(args: argparse.Namespace) -> Path:
    if args.demo_package is not None:
        return save_demo_package_to_rrd(
            args.demo_package, args.output, max_points=args.max_points
        )
    if args.dense_scene is not None:
        return save_dense_scene_to_rrd(
            args.dense_scene,
            args.output,
            max_points=args.max_points,
            use_ransac=not args.no_ransac,
        )
    if args.colmap_model is not None:
        return save_trajectory_comparison_to_rrd(
            args.tracks_npz,
            args.colmap_model,
            args.output,
            use_ransac=not args.no_ransac,
        )
    return save_static_tracks_to_rrd(
        args.tracks_npz, args.output, max_points=args.max_points
    )


def _spawn_viewer(args: argparse.Namespace) -> None:
    if args.demo_package is not None:
        visualize_demo_package(args.demo_package, max_points=args.max_points)
    elif args.dense_scene is not None:
        visualize_dense_scene(
            args.dense_scene, max_points=args.max_points, use_ransac=not args.no_ransac
        )
    elif args.colmap_model is not None:
        visualize_trajectory_comparison(
            args.tracks_npz, args.colmap_model, use_ransac=not args.no_ransac
        )
    else:
        visualize_static_tracks(args.tracks_npz, max_points=args.max_points)


# ---------------------------------------------------------------------------
# Screenshot mode: rerun --serve-web + Playwright CLI (headless).
# ---------------------------------------------------------------------------


def _viewer_url(web_port: int, grpc_port: int) -> str:
    data_url = f"rerun+http://localhost:{grpc_port}/proxy"
    return f"http://127.0.0.1:{web_port}?url={urllib.parse.quote(data_url, safe='')}"


def _wait_for_tcp_port(
    host: str, port: int, timeout_s: float, proc: subprocess.Popen, log: TextIO
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log.flush()
            log.seek(0)
            raise RuntimeError(
                f"rerun exited before serving {port}:\n{log.read()[-2000:]}"
            )
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def _playwright_cli_screenshot(url: str, png_path: Path, render_wait: float) -> None:
    playwright = shutil.which("playwright")
    if playwright is None:
        raise RuntimeError("playwright CLI not found; run `uv sync --extra dev`")
    subprocess.run(
        [
            playwright,
            "screenshot",
            "--browser",
            "chromium",
            "--viewport-size",
            SCREENSHOT_VIEWPORT,
            "--wait-for-timeout",
            str(int(render_wait * 1000)),
            "--timeout",
            SCREENSHOT_TIMEOUT_MS,
            url,
            str(png_path),
        ],
        check=True,
    )


def screenshot_rrd(rrd_path: Path, png_path: Path, args: argparse.Namespace) -> Path:
    """Serve ``rrd_path`` with ``rerun --serve-web`` and screenshot it."""
    if not rrd_path.is_file():
        raise FileNotFoundError(rrd_path)
    if shutil.which("rerun") is None:
        raise RuntimeError(
            "rerun CLI not found; install rerun-sdk (`uv sync --extra vis`)"
        )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile("w+", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [
                "rerun",
                "--serve-web",
                "--web-viewer-port",
                str(args.web_port),
                "--port",
                str(args.grpc_port),
                str(rrd_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_tcp_port(
                "127.0.0.1", args.web_port, args.serve_timeout, proc, log
            )
            _playwright_cli_screenshot(
                _viewer_url(args.web_port, args.grpc_port), png_path, args.render_wait
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    return png_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.colmap_model is not None and args.tracks_npz is None:
        raise SystemExit("--colmap-model requires --tracks-npz")

    if args.mode == "viewer":
        _spawn_viewer(args)
        return 0

    rrd_path = _write_rrd(args)
    print(f"Wrote rerun recording to {rrd_path}")
    if args.mode == "screenshot":
        png = screenshot_rrd(rrd_path, args.screenshot, args)
        print(f"Wrote screenshot to {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
