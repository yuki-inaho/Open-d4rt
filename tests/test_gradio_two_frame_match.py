from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import gradio_two_frame_match as matcher


def test_prepare_two_frame_video_resizes_each_input() -> None:
    source = np.zeros((12, 20, 3), dtype=np.uint8)
    target = np.zeros((20, 10, 3), dtype=np.uint8)

    video = matcher.prepare_two_frame_video(source, target, (8, 16))

    assert video.shape == (2, 8, 16, 3)
    assert video.dtype == np.uint8


def test_prediction_to_matches_filters_visibility_and_out_of_bounds() -> None:
    matches = matcher.prediction_to_matches(
        source_xy=np.asarray([[4, 6], [8, 10], [12, 14]], dtype=np.float32),
        target_uv_norm=np.asarray(
            [[0.5, 0.25], [1.1, 0.4], [0.2, 0.7]], dtype=np.float32
        ),
        visibility_logits=np.asarray([2.0, 5.0, -5.0], dtype=np.float32),
        confidence=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        target_hw=(21, 41),
        visibility_threshold=0.5,
    )

    assert matches.source_xy.tolist() == [[4.0, 6.0]]
    assert np.allclose(matches.target_xy, [[20.0, 5.0]])
    assert matches.visibility_probability[0] > 0.8


def test_overlay_and_rows_keep_original_coordinates() -> None:
    source = np.zeros((10, 12, 3), dtype=np.uint8)
    target = np.zeros((14, 16, 3), dtype=np.uint8)
    matches = matcher.MatchSet(
        source_xy=np.asarray([[3.0, 4.0]], dtype=np.float32),
        target_xy=np.asarray([[8.0, 9.0]], dtype=np.float32),
        visibility_probability=np.asarray([0.9], dtype=np.float32),
        confidence=np.asarray([0.7], dtype=np.float32),
    )

    overlay = matcher.render_match_overlay(source, target, matches)
    rows = matcher.match_rows(matches)

    assert overlay.shape == (14, 12 + 28 + 16, 3)
    assert np.allclose(rows, [[3.0, 4.0, 8.0, 9.0, 0.9, 0.7]])


def test_build_ui_does_not_load_model() -> None:
    settings = matcher.AppSettings(
        config_path=matcher.REPO_ROOT / "missing.yaml",
        checkpoint_path=matcher.REPO_ROOT / "missing.ckpt",
        device="cpu",
    )

    ui = matcher.build_ui(settings)

    assert ui.__class__.__name__ == "Blocks"
