from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import visualize_rerun as cli
from tests.test_check_colmap_trajectory_consistency import (
    _build_synthetic_scene,
    _write_colmap_text,
    _write_pred_npz,
)


def test_cli_rejects_both_inputs(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--demo-package", str(tmp_path), "--tracks-npz", str(tmp_path)])


def test_cli_requires_an_input() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--mode", "rrd"])


def test_cli_colmap_model_requires_tracks_npz(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="requires --tracks-npz"):
        cli.main(["--demo-package", str(tmp_path), "--colmap-model", str(tmp_path)])


def test_cli_rrd_mode_writes_recording_for_tracks(tmp_path: Path) -> None:
    k, pts, uv, names, _, _, _, _, _ = _build_synthetic_scene()
    npz = tmp_path / "pred.npz"
    _write_pred_npz(npz, k, pts, uv, names)
    rrd = tmp_path / "out.rrd"

    rc = cli.main(["--tracks-npz", str(npz), "--mode", "rrd", "--output", str(rrd)])

    assert rc == 0
    assert rrd.stat().st_size > 1024


@pytest.mark.skipif(
    os.environ.get("D4RT_RUN_SCREENSHOT_TESTS") != "1",
    reason="set D4RT_RUN_SCREENSHOT_TESTS=1 to run the rerun+playwright screenshot E2E",
)
def test_cli_screenshot_mode_writes_png(tmp_path: Path) -> None:
    k, pts, uv, names, _, _, _, colmap_r, colmap_t = _build_synthetic_scene()
    model = tmp_path / "model"
    _write_colmap_text(model, k, names, colmap_r, colmap_t)
    npz = tmp_path / "pred.npz"
    _write_pred_npz(npz, k, pts, uv, names)
    rrd = tmp_path / "out.rrd"
    png = tmp_path / "out.png"

    rc = cli.main(
        [
            "--tracks-npz",
            str(npz),
            "--colmap-model",
            str(model),
            "--mode",
            "screenshot",
            "--output",
            str(rrd),
            "--screenshot",
            str(png),
            "--web-port",
            "9133",
            "--grpc-port",
            "9134",
            "--render-wait",
            "3.0",
        ]
    )

    assert rc == 0
    assert png.stat().st_size > 1024
