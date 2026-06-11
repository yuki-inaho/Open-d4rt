from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_demo_from_video as demo


def test_positive_int_rejects_zero_and_negative_values() -> None:
    assert demo._positive_int("3") == 3
    with pytest.raises(argparse.ArgumentTypeError):
        demo._positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        demo._positive_int("-1")


def test_non_negative_float_accepts_zero_for_input_fps_passthrough() -> None:
    assert demo._non_negative_float("0") == 0.0
    assert demo._non_negative_float("12.5") == 12.5
    with pytest.raises(argparse.ArgumentTypeError):
        demo._non_negative_float("-0.1")


def test_load_frames_decodes_gif_fallback_and_respects_max_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gif_path = tmp_path / "clip.gif"
    frames = [
        Image.fromarray(np.full((8, 10, 3), fill_value=idx * 40, dtype=np.uint8))
        for idx in range(3)
    ]
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0
    )

    def fail_video_decode(_path: Path, _max_frames: int) -> tuple[np.ndarray, float]:
        raise RuntimeError("force GIF fallback")

    monkeypatch.setattr(demo, "_load_video_rgb", fail_video_decode)

    video_rgb, fps = demo._load_frames(gif_path, max_frames=2)

    assert video_rgb.shape == (2, 8, 10, 3)
    assert video_rgb.dtype == np.uint8
    assert fps == pytest.approx(10.0)


def test_write_demo_package_emits_viewer_assets(tmp_path: Path) -> None:
    t_count = 3
    h, w = 16, 20
    point_count = 4
    track_count = 2
    video_rgb = np.zeros((t_count, h, w, 3), dtype=np.uint8)
    for frame_idx in range(t_count):
        video_rgb[frame_idx, :, :, 0] = 30 + frame_idx * 30
        video_rgb[frame_idx, :, :, 1] = np.arange(w, dtype=np.uint8)[None, :]
        video_rgb[frame_idx, :, :, 2] = np.arange(h, dtype=np.uint8)[:, None]

    point_query_uv = np.array(
        [[1.0, 1.0], [5.0, 4.0], [10.0, 8.0], [18.0, 14.0]],
        dtype=np.float32,
    )
    point_xyz = np.stack(
        [
            np.column_stack(
                [
                    np.arange(point_count),
                    np.full(point_count, frame_idx),
                    np.ones(point_count),
                ]
            )
            for frame_idx in range(t_count)
        ],
        axis=0,
    ).astype(np.float32)
    track_xyz = np.stack(
        [
            np.column_stack(
                [
                    np.full(t_count, track_idx),
                    np.arange(t_count),
                    np.ones(t_count) * (track_idx + 1),
                ]
            )
            for track_idx in range(track_count)
        ],
        axis=0,
    ).astype(np.float32)

    package = {
        "num_frames": t_count,
        "video_width": w,
        "video_height": h,
        "clip_frames": 48,
        "track_stitch_diagnostics": {},
        "track_query_uv_px": point_query_uv[:track_count],
        "track_query_t_src": np.zeros((track_count,), dtype=np.int64),
        "track_xyz_ref0": track_xyz,
        "track_uv_px": np.tile(
            point_query_uv[:track_count, None, :],
            (1, t_count, 1),
        ).astype(np.float32),
        "track_visibility": np.ones((track_count, t_count), dtype=bool),
        "track_confidence": np.ones((track_count, t_count), dtype=np.float32),
        "point_query_uv_px": point_query_uv,
        "point_xyz_ref0": point_xyz,
        "point_visibility": np.ones((t_count, point_count), dtype=bool),
        "point_uv_px": np.tile(point_query_uv[None, :, :], (t_count, 1, 1)).astype(
            np.float32
        ),
        "point_confidence": np.ones((t_count, point_count), dtype=np.float32),
        "point_motion_score": np.linspace(0.0, 1.0, num=point_count, dtype=np.float32),
        "point_is_dynamic": np.array([0, 0, 1, 1], dtype=bool),
        "point_rgb": np.tile(
            np.array(
                [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]],
                dtype=np.uint8,
            ),
            (t_count, 1, 1),
        ),
        "bounds_min": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "bounds_max": np.array([3.0, 2.0, 2.0], dtype=np.float32),
        "bounds_center": np.array([1.5, 1.0, 1.0], dtype=np.float32),
        "bounds_radius": np.array([2.0], dtype=np.float32),
        "ref0_K": np.array(
            [[12.0, 0.0, w / 2.0], [0.0, 12.0, h / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
    }

    output_dir = tmp_path / "demo_package"
    input_path = tmp_path / "source.gif"
    input_path.write_bytes(b"placeholder")

    demo._write_demo_package(
        output_dir=output_dir,
        input_path=input_path,
        package=package,
        video_rgb=video_rgb,
        fps=6.0,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    data = json.loads((output_dir / "assets" / "demo_data.json").read_text())

    assert manifest["viewer"] == "viser"
    assert manifest["data_json"] == "assets/demo_data.json"
    assert data["meta"]["numFrames"] == t_count
    assert data["meta"]["videoWidth"] == w
    assert data["meta"]["trackCount"] == track_count
    assert len(data["points"]["xyzRef0"]) == t_count
    assert len(data["points"]["xyzRef0"][0]) == point_count
    assert len(data["tracks"]["xyzRef0"]) == track_count
    assert len(data["tracks"]["xyzRef0"][0]) == t_count

    video_path = output_dir / "assets" / "input_video.mp4"
    poster_path = output_dir / "assets" / "video_poster.jpg"
    assert video_path.exists()
    assert poster_path.exists()

    cap = cv2.VideoCapture(str(video_path))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == t_count
    finally:
        cap.release()
