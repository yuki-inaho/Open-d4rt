#!/usr/bin/env python3
"""Build a Viser demo package from a local video or GIF."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageSequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_track_3d import _load_video_rgb, _resolve_device, _resize_video, _unwrap_state_dict
from src.core import build_logger, load_yaml_config, seed_everything
from src.model import build_model
from vis.build_like_demo import (
    _build_uv_grid,
    _export_demo_data,
    _export_video_from_frames,
    _jsonable_float_array,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lightweight Viser demo package from a local video or GIF.")
    parser.add_argument("--config", required=True, help="Model config YAML.")
    parser.add_argument("--ckpt-path", required=True, help="OpenD4RT checkpoint path.")
    parser.add_argument("--input", required=True, help="Input video/GIF path.")
    parser.add_argument("--output-dir", required=True, help="Output demo package directory.")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--point-grid-cols", type=int, default=8)
    parser.add_argument("--point-grid-rows", type=int, default=8)
    parser.add_argument("--point-max-points", type=int, default=64)
    parser.add_argument("--track-max-points", type=int, default=16)
    parser.add_argument("--track-min-visible-frames", type=int, default=2)
    parser.add_argument("--point-query-chunk-size", type=int, default=16)
    parser.add_argument("--track-query-chunk-size", type=int, default=16)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--no-suppress-depth-boundary-tracks", action="store_true")
    return parser.parse_args()


def _load_frames(path: Path, max_frames: int) -> tuple[np.ndarray, float]:
    try:
        return _load_video_rgb(path, max_frames=max_frames)
    except Exception:
        if path.suffix.lower() != ".gif":
            raise

    frames: list[np.ndarray] = []
    with Image.open(path) as img:
        duration_ms = float(img.info.get("duration", 125) or 125)
        fps = 1000.0 / max(duration_ms, 1.0)
        for frame in ImageSequence.Iterator(img):
            rgb = frame.convert("RGB")
            frames.append(np.asarray(rgb, dtype=np.uint8))
            if max_frames > 0 and len(frames) >= int(max_frames):
                break
    if not frames:
        raise RuntimeError(f"No frames decoded from GIF: {path}")
    return np.stack(frames, axis=0), float(fps)


def _load_checkpoint_model(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", mmap=True)
    state = _unwrap_state_dict(payload)
    if not state:
        raise RuntimeError(f"No model weights found in checkpoint: {path}")
    return state


def _write_demo_package(
    *,
    output_dir: Path,
    input_path: Path,
    package: dict[str, Any],
    video_rgb: np.ndarray,
    fps: float,
) -> None:
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "fps": float(fps if fps > 0.0 else 8.0),
        "numFrames": int(package["num_frames"]),
        "videoWidth": int(package["video_width"]),
        "videoHeight": int(package["video_height"]),
        "crop": {"top": 0, "bottom": int(package["video_height"])},
        "clipFrames": int(package["clip_frames"]),
        "trackStitchDiagnostics": package.get("track_stitch_diagnostics", {}),
        "trackCount": int(package["track_query_uv_px"].shape[0]),
        "trackCountPred": int(package["track_query_uv_px"].shape[0]),
        "trackCountGt": 0,
        "pointCountPerFrame": int(package["point_query_uv_px"].shape[0]),
        "bounds": {
            "min": _jsonable_float_array(package["bounds_min"]),
            "max": _jsonable_float_array(package["bounds_max"]),
            "center": _jsonable_float_array(package["bounds_center"]),
            "radius": float(package["bounds_radius"][0]),
        },
        "ref0K": _jsonable_float_array(package["ref0_K"], ndigits=5),
        "camera": None,
        "cameraPred": None,
        "depthPred": None,
        "source": {
            "type": "local_video",
            "path": str(input_path),
        },
    }

    data_json = {
        "meta": meta,
        "tracks": {
            "queryUvPx": _jsonable_float_array(package["track_query_uv_px"], ndigits=3),
            "queryTSrc": package["track_query_t_src"].astype(np.int32).tolist(),
            "xyzRef0": _jsonable_float_array(package["track_xyz_ref0"], ndigits=5),
            "uvPx": _jsonable_float_array(package["track_uv_px"], ndigits=3),
            "visibility": package["track_visibility"].astype(np.int32).tolist(),
            "confidence": _jsonable_float_array(package["track_confidence"], ndigits=4),
        },
        "points": {
            "queryUvPx": _jsonable_float_array(package["point_query_uv_px"], ndigits=3),
            "xyzRef0": _jsonable_float_array(package["point_xyz_ref0"], ndigits=5),
            "visibility": package["point_visibility"].astype(np.int32).tolist(),
            "rgb": package["point_rgb"].astype(np.int32).tolist(),
            "uvPx": _jsonable_float_array(package["point_uv_px"], ndigits=3),
            "confidence": _jsonable_float_array(package["point_confidence"], ndigits=4),
            "motionScore": _jsonable_float_array(package["point_motion_score"], ndigits=5),
            "isDynamic": np.asarray(package["point_is_dynamic"], dtype=np.int32).tolist(),
        },
        "pointsRaw": {
            "xyzRef0": _jsonable_float_array(package["point_xyz_ref0"], ndigits=5),
        },
    }

    (assets_dir / "demo_data.json").write_text(json.dumps(data_json, ensure_ascii=False), encoding="utf-8")
    video_name, poster_name = _export_video_from_frames(
        video_rgb=video_rgb,
        fps=float(fps if fps > 0.0 else 8.0),
        dst_video=assets_dir / "input_video.mp4",
    )
    manifest = {
        "input": str(input_path),
        "video_copy": f"assets/{video_name}",
        "video_poster": f"assets/{poster_name}",
        "data_json": "assets/demo_data.json",
        "viewer": "viser",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger("build_demo_from_video", output_dir)

    cfg = load_yaml_config(args.config)
    seed_everything(int(cfg.get_path("experiment.seed", 42)), deterministic=True)

    input_path = Path(args.input).resolve()
    ckpt_path = Path(args.ckpt_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    video_rgb, input_fps = _load_frames(input_path, max_frames=int(args.num_frames))
    fps = float(args.fps if args.fps > 0.0 else input_fps)
    image_size = cfg.get_path("model.input.image_size", [256, 256])
    video_model_rgb = _resize_video(video_rgb, image_hw=(int(image_size[0]), int(image_size[1])))

    point_query_uv_px = _build_uv_grid(
        width=int(video_rgb.shape[2]),
        height=int(video_rgb.shape[1]),
        cols=int(args.point_grid_cols),
        rows=int(args.point_grid_rows),
        max_points=int(args.point_max_points),
    )

    logger.info("Building model from %s", args.config)
    model = build_model(cfg["model"]).eval()
    state = _load_checkpoint_model(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    del state
    gc.collect()

    device = _resolve_device(args.device)
    model = model.to(device).eval()
    logger.info("Running inference on %s with %d frames and %d point queries", device, video_rgb.shape[0], point_query_uv_px.shape[0])
    package = _export_demo_data(
        model=model,
        video_rgb=video_rgb,
        video_model_rgb=video_model_rgb,
        point_query_uv_px=point_query_uv_px,
        point_query_chunk_size=int(args.point_query_chunk_size),
        track_query_chunk_size=int(args.track_query_chunk_size),
        track_selection="motion",
        track_max_points=int(args.track_max_points),
        track_min_visible_frames=int(args.track_min_visible_frames),
        suppress_depth_boundary_tracks=not bool(args.no_suppress_depth_boundary_tracks),
    )
    _write_demo_package(output_dir=output_dir, input_path=input_path, package=package, video_rgb=video_rgb, fps=fps)
    logger.info("Saved demo package to %s", output_dir)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
